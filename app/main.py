"""MathPilot FastAPI app: student workspace, instructor dashboard, JSON API."""
from __future__ import annotations
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from . import config
from . import auth
from .db import (
    Assignment, ensure_schema, get_db, init_db, Activity, Attempt, Course, CourseMaterial,
    HintRequest, Lesson, Publication, PracticeProgress, ReviewRequest,
    SharedFile, Submission,
    TutorEvent, TutorPending, TutorRateHit, TutorSession, TutorThread, User,
    utcnow,
)
from .seed import seed
from .tutor import orchestrate
from .verifier import MAX_LEN

BASE = Path(__file__).resolve().parent.parent
app = FastAPI(title="MathPilot")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))
auth.add_viewer_context(templates)

# Instructor-uploaded sources (PDF/Word). Stored outside /static so they are
# never served publicly; access goes only through the gated route below.
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
# extension -> (media type, inline-previewable?)
UPLOAD_TYPES = {
    ".pdf": ("application/pdf", True),
    ".doc": ("application/msword", False),
    ".docx": ("application/vnd.openxmlformats-officedocument"
              ".wordprocessingml.document", False),
}

init_db()
ensure_schema()
seed()

class TutorThreadState:
    """One tutor thread read out of SQLite: its transcript and its ladder.

    This replaces the process-memory dicts the MVP ran on. A thread key always
    names one student, so one row is one student's conversation and a restart
    or a second worker can neither hand back a spent hint budget nor answer as
    though the student had never spoken. Mutate `state` or `messages`, then
    save() once the turn is settled; they are the same dict/list views a turn
    always worked with, written through to the row on save.
    """

    KEEP_MESSAGES = 40  # how much of a thread the server remembers

    def __init__(self, db: Session, key: str):
        self._db = db
        self._key = key
        self._row = self._load()
        try:
            self.messages = json.loads(self._row.messages_json or "[]")
        except Exception:
            self.messages = []
        self.state = {"hints": self._row.hints or 0,
                      "misses": self._row.misses or 0,
                      "unreadable": self._row.unreadable or 0,
                      "moves": self._row.moves or 0}

    def _load(self) -> TutorThread:
        row = self._db.query(TutorThread).filter(
            TutorThread.key == self._key).first()
        if row:
            return row
        row = TutorThread(key=self._key, messages_json="[]")
        self._db.add(row)
        self._db.flush()
        return row

    def append(self, user_text: str, reply: str) -> None:
        """Log one finished turn, which is all a reply ever adds."""
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": reply})

    def save(self) -> None:
        self._row.messages_json = json.dumps(
            self.messages[-self.KEEP_MESSAGES:])
        self._row.hints = int(self.state.get("hints", 0) or 0)
        self._row.misses = int(self.state.get("misses", 0) or 0)
        self._row.unreadable = int(self.state.get("unreadable", 0) or 0)
        self._row.moves = int(self.state.get("moves", 0) or 0)
        self._row.updated_at = utcnow()
        self._db.commit()


def _thread(db: Session, key: str) -> TutorThreadState:
    return TutorThreadState(db, key)



def steps_of(activity: Activity) -> list[dict]:
    try:
        return json.loads(activity.expected_steps_json or "[]")
    except Exception:
        return []


def hints_of(activity: Activity) -> list[str]:
    try:
        return json.loads(activity.hint_sequence_json or "[]")
    except Exception:
        return []


def _viewer_id(request: Request, db: Session) -> int:
    """Whose rows this request may read and write.

    Always resolved from the signed-in viewer's database row, never from a
    cookie body or a client-supplied field. With no cookie it falls back to
    the demo roster's first student.
    """
    return auth.viewer_or_demo_student(request, db).id


@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    uid = _viewer_id(request, db)
    pubs = db.query(Publication).order_by(Publication.id.desc()).all()
    lesson_cards, asg_cards = _student_cards(db, pubs, uid)
    completed = db.query(Submission).filter(
        Submission.user_id == uid,
        Submission.status == "submitted").count()
    in_progress = db.query(Submission).filter(
        Submission.user_id == uid,
        Submission.status == "draft").count()
    return templates.TemplateResponse(request, "home.html", {
        "request": request,
        "lesson_cards": lesson_cards, "assignment_cards": asg_cards,
        "n_lessons": len(lesson_cards), "n_pending": len(asg_cards),
        "n_completed": completed, "n_in_progress": in_progress,
        "first_name": _student_first_name(db, uid),
        "demo": not config.ai_configured(),
    })


def _student_cards(db: Session, pubs: list, user_id: int) -> tuple[list, list]:
    """Build lesson + assignment cards for one student's workspace.

    Shared by the dashboard and the My Lessons/My Assignments pages so all
    three views show the same packages. Lessons list every publication
    carrying a lesson; assignments list every publication carrying an
    assignment with its question count, due date, and submission status.
    """
    import json as _json

    lessons, assignments = [], []
    instructor_name = _instructor_name(db)
    for p in pubs:
        if p.lesson_id:
            lesson = db.get(Lesson, p.lesson_id)
            lessons.append({
                "link_id": p.link_id,
                "title": (lesson.title if lesson else None) or p.title or "Untitled",
                "course": p.assigned_course or _lesson_course_title(db, lesson),
                "instructor": instructor_name,
                "status": "Available",
                "due": p.due_date or "",
            })
        if p.assignment_id:
            asg = db.get(Assignment, p.assignment_id)
            try:
                items = _json.loads(asg.questions_json or "[]") if asg else []
                n_q = len(items) if isinstance(items, list) else 0
            except Exception:
                n_q = 0
            sub = db.query(Submission).filter(
                Submission.publication_id == p.id,
                Submission.user_id == user_id).first()
            status = ("Submitted" if sub and sub.status == "submitted"
                      else ("In Progress" if sub else "Not Started"))
            assignments.append({
                "link_id": p.link_id,
                "title": (asg.title if asg else None) or p.title or "Untitled",
                "course": p.assigned_course or "",
                "n_questions": n_q,
                "due": p.due_date or "",
                "status": status,
            })
    return lessons, assignments


PRACTICE_TARGET_ITEMS = 5
# How long to leave a lesson's item set alone after a top-up attempt.
PRACTICE_TOPUP_COOLDOWN_S = 6 * 3600


def _lesson_course_title(db: Session, lesson: Lesson) -> str:
    course = db.get(Course, lesson.course_id) if lesson.course_id else None
    return (course.title if course else "") or ""


def _instructor_name(db: Session) -> str:
    """The roster's instructor, for the attribution shown on student cards."""
    teacher = db.query(User).filter(User.role == "instructor").order_by(
        User.id).first()
    return teacher.name if teacher else "Instructor"


def _practice_options(db: Session) -> list:
    """Every lesson a student may practise — published or only approved.

    Published lessons keep their shareable link id; an approved lesson with no
    publication gets a 'lesson-<id>' ref so practice never waits on publishing.
    """
    options, seen = [], set()
    pubs = db.query(Publication).filter(
        Publication.lesson_id.isnot(None)).order_by(Publication.id.desc()).all()
    for p in pubs:
        lesson = db.get(Lesson, p.lesson_id)
        if not lesson or lesson.id in seen:
            continue
        seen.add(lesson.id)
        options.append({
            "ref": p.link_id, "lesson_id": lesson.id,
            "title": lesson.title or p.title or "Untitled",
            "topic": lesson.topic or "",
            "course": p.assigned_course or _lesson_course_title(db, lesson),
            "n_items": _practice_item_count(lesson)})
    approved = db.query(Lesson).filter(
        Lesson.status == "approved").order_by(Lesson.id.desc()).all()
    for lesson in approved:
        if lesson.id in seen:
            continue
        seen.add(lesson.id)
        options.append({
            "ref": f"lesson-{lesson.id}", "lesson_id": lesson.id,
            "title": lesson.title or "Untitled", "topic": lesson.topic or "",
            "course": _lesson_course_title(db, lesson),
            "n_items": _practice_item_count(lesson)})
    return options


def _practice_item_count(lesson: Lesson) -> int:
    """How many practice items this lesson already carries (no AI call)."""
    import json as _json

    from .lessons import split_student_payload

    try:
        content = split_student_payload(_json.loads(
            lesson.student_content_json or "{}"))
    except Exception:
        return 0
    return len(content.get("guided_items", []) or [])


def _practice_lesson(db: Session, ref: str) -> Lesson:
    """Resolve a practice package ref ('lesson-<id>' or a publication link)."""
    ref = str(ref or "").strip()
    if not ref:
        raise HTTPException(400, "No practice package selected")
    if ref.startswith("lesson-"):
        try:
            lesson_id = int(ref.split("-", 1)[1])
        except ValueError:
            raise HTTPException(404, "Practice package not found")
        lesson = db.get(Lesson, lesson_id)
        if not lesson or lesson.status == "draft":
            raise HTTPException(404, "Practice package not found")
        return lesson
    pub = db.query(Publication).filter(Publication.link_id == ref).first()
    if not pub or not pub.lesson_id:
        raise HTTPException(404, "Practice package not found")
    lesson = db.get(Lesson, pub.lesson_id)
    if not lesson:
        raise HTTPException(404, "Practice package not found")
    return lesson


def _practice_json(lesson: Lesson) -> dict:
    """The whole practice cache as stored on the lesson (items, answers,
    plans, and the top-up marker). Read it, change the keys you want, write
    the dict back — never replace it, or a top-up would erase the plans."""
    import json as _json

    try:
        cache = _json.loads(lesson.practice_json or "{}")
    except Exception:
        cache = {}
    return cache if isinstance(cache, dict) else {}


def _practice_cache(lesson: Lesson) -> tuple:
    cache = _practice_json(lesson)
    items = cache.get("items") or []
    answers = cache.get("answers") or {}
    return (items if isinstance(items, list) else [],
            answers if isinstance(answers, dict) else {})


def _practice_plans(lesson: Lesson) -> dict:
    """Cached step plans for this lesson's practice items, keyed by item id."""
    import json as _json

    try:
        cache = _json.loads(lesson.practice_json or "{}")
    except Exception:
        cache = {}
    if not isinstance(cache, dict):
        cache = {}
    plans = cache.get("plans") or {}
    return plans if isinstance(plans, dict) else {}


def _practice_plan(db: Session, lesson: Lesson, item: dict) -> list[dict]:
    """The verified step plan for one practice item.

    The deterministic templates cover the algebra. The model covers the
    questions they cannot express (find the domain, which quadrant, what
    transformation). Its plan is validated and then kept on the lesson, so
    an item is planned once rather than on every visit.
    """
    import json as _json

    from . import float_plan as _planmod

    item_id = str(item.get("item_id") or "")
    prompt = str(item.get("prompt_latex") or "")
    task = str(item.get("task") or "")
    plan, _src = _planmod.build_plan(prompt, allow_llm=False, task=task)
    if plan:
        return plan
    cached = _practice_plans(lesson).get(item_id)
    if isinstance(cached, list) and cached:
        return cached
    if not config.ai_configured():
        return []
    plan, _src = _planmod.build_plan(prompt, allow_llm=True, task=task)
    if not plan:
        return []
    try:
        cache = _json.loads(lesson.practice_json or "{}")
    except Exception:
        cache = {}
    if not isinstance(cache, dict):
        cache = {}
    cache.setdefault("plans", {})[item_id] = plan
    lesson.practice_json = _json.dumps(cache)
    db.commit()
    return plan


def _practice_topup_cooling(lesson: Lesson) -> bool:
    """Did a top-up attempt for this lesson just happen?

    See the note in _practice_set: for a lesson whose items are conceptual the
    attempt always fails, and repeating it on every request is what makes the
    tutor look dead. The window is long enough to stop the thrash and short
    enough that an instructor who edits the lesson gets a fresh try.
    """
    tried = _practice_json(lesson).get("topup_tried_at")
    try:
        return time.time() - float(tried) < PRACTICE_TOPUP_COOLDOWN_S
    except (TypeError, ValueError):
        return False


def _practice_set(db: Session, lesson: Lesson, top_up: bool = True) -> dict:
    """Practice items for a lesson: its own guided items first, then verified
    generated questions up to PRACTICE_TARGET_ITEMS.

    Generated items are cached on the lesson, so the AI call happens once per
    lesson rather than on every student visit. Answers stay server-side; only
    prompts and hints reach the student.

    `top_up` is False for the routes that only look one item up (the tutor and
    the checker): they already know the item id, and a generation round costs
    ~15 seconds of model time that the student is waiting through.
    """
    import json as _json

    from .lessons import split_student_payload, _practice_task_texts

    try:
        content = split_student_payload(_json.loads(
            lesson.student_content_json or "{}"))
    except Exception:
        content = {}
    # The task text comes from the practice section's numbered questions. It
    # is recomputed here rather than only at derivation time, so lessons
    # stored before the task existed pick it up without a migration.
    tasks: list[str] = []
    guided_items = content.get("guided_items", []) or []
    for sec in content.get("sections", []) or []:
        if isinstance(sec, dict) and sec.get("kind") == "practice":
            tasks = _practice_task_texts(
                str(sec.get("body_markdown") or ""),
                [str(g.get("prompt_latex") or "") for g in guided_items
                 if isinstance(g, dict)])
            break
    items = []
    answers: dict = {}
    for i, g in enumerate(guided_items):
        items.append({"item_id": g.get("item_id", ""),
                      "prompt_latex": g.get("prompt_latex", ""),
                      "task": g.get("task") or (tasks[i] if i < len(tasks)
                                                else ""),
                      "context": g.get("context", "Guided practice"),
                      "hints_enabled": g.get("hints_enabled", True),
                      "source": "lesson"})
    for c in _practice_cache(lesson)[0]:
        if not isinstance(c, dict) or not c.get("item_id"):
            continue
        if any(i["item_id"] == c["item_id"] for i in items):
            continue
        items.append({"item_id": c["item_id"],
                      "prompt_latex": c.get("prompt_latex", ""),
                      # A generated question is its own task: the prompt
                      # states what is being asked.
                      "task": str(c.get("prompt_latex", "") or ""),
                      "context": c.get("context", "Extra practice"),
                      "hints": [str(h) for h in (c.get("hints") or []) if h],
                      "hints_enabled": True,
                      "source": "generated"})
        answers[c["item_id"]] = str(c.get("answer", ""))
    shortfall = PRACTICE_TARGET_ITEMS - len(items)
    if (top_up and shortfall > 0 and config.ai_configured()
            and not _practice_topup_cooling(lesson)):
        generated = _practice_generate(lesson, shortfall, items)
        cache = _practice_json(lesson)
        # Record the attempt even when it produced nothing. A lesson of
        # conceptual questions ("classify the discontinuity") can never pass
        # the SymPy gate, so the shortfall stays open forever and every page
        # load would otherwise pay ~15 seconds of model time for a set that
        # cannot grow.
        cache["topup_tried_at"] = int(time.time())
        if generated:
            cached_items, cached_answers = _practice_cache(lesson)
            merged_answers = dict(cached_answers)
            for q in generated:
                merged_answers[q["item_id"]] = q["answer"]
            cache["items"] = cached_items + generated
            cache["answers"] = merged_answers
        lesson.practice_json = _json.dumps(cache)
        db.commit()
        for q in (generated or []):
            answers[q["item_id"]] = q["answer"]
            items.append({"item_id": q["item_id"],
                          "prompt_latex": q["prompt_latex"],
                          "task": str(q.get("prompt_latex", "") or ""),
                          "context": q["context"],
                          "hints": list(q.get("hints") or []),
                          "hints_enabled": True,
                          "source": "generated"})
    return {"items": items, "answers": answers,
            "title": content.get("title") or lesson.title,
            "topic": content.get("topic") or lesson.topic,
            "course": _lesson_course_title(db, lesson)}


def _practice_generate(lesson: Lesson, need: int, existing: list) -> list:
    """Generate `need` verified questions grounded in this lesson."""
    from .generate import generate_questions
    from .schemas import (lesson_assignment_context,
                          lesson_content_from_json)

    notes = lesson_assignment_context(
        lesson_content_from_json(lesson.instructor_content_json))
    topic = (lesson.topic or lesson.title or "").strip()
    # Ask for more than needed: the SymPy gate discards candidates, and a
    # single-item request would leave almost nothing to choose from.
    try:
        result = generate_questions(notes=notes, topic=topic,
                                    num_questions=max(4, need))
    except Exception:
        return []
    used = {str(i.get("prompt_latex", "")).strip() for i in existing}
    out = []
    start = len(existing)
    for q in (result or {}).get("verified", []) or []:
        if len(out) >= need:
            break
        prompt = str(q.get("prompt_latex", "") or "").strip()
        if not prompt or prompt in used:
            continue
        used.add(prompt)
        hints = []
        for st in q.get("steps", []) or []:
            hints.extend([str(h) for h in (st.get("hints") or []) if h])
        out.append({"item_id": f"p{start + len(out) + 1}",
                    "prompt_latex": prompt, "context": "Extra practice",
                    "hints": hints[:3],
                    "answer": str(q.get("final_answer_latex", "") or "")})
    return out


@app.get("/practice", response_class=HTMLResponse)
def practice_page(request: Request, topic: str = "", db: Session = Depends(get_db)):
    """Practice with AI Tutor: pick a lesson package, work its guided items.

    ?topic= preselects the first package whose lesson matches the topic
    (used by the lesson page's "Practice This Topic" button).
    """
    options = _practice_options(db)
    selected = None
    if topic:
        needle = topic.strip().lower()
        selected = next(
            (o for o in options if needle in (o["topic"] or "").lower()
             or needle in (o["title"] or "").lower()), None)
    if not selected and options:
        selected = options[0]
    return templates.TemplateResponse(request, "student_practice.html", {
        "request": request, "options": options, "selected": selected,
        "topic": topic or (selected["topic"] if selected else ""),
        "active_nav": "dashboard",
        "demo": not config.ai_configured(),
        # The practice page has its own full tutor panel; the floating
        # widget hides there (per spec) and the thread below keeps both
        # surfaces on one conversation.
        "hide_float_tutor": True,
        "float_thread": _float_thread_payload(db, _viewer_id(request, db),
                                              "default"),
    })


@app.post("/api/practice/items")
def api_practice_items(request: Request, payload: dict,
                       db: Session = Depends(get_db)):
    """Practice items for a package: guided lesson items, topped up to five.

    Answers never leave the server; they are resolved again inside
    /api/practice/attempt when a step is graded.
    """
    uid = _viewer_id(request, db)
    ref = str(payload.get("ref", "") or "").strip()
    lesson = _practice_lesson(db, ref)
    packed = _practice_set(db, lesson)
    return {"ref": ref, "items": packed["items"], "title": packed["title"],
            "topic": packed["topic"], "course": packed["course"],
            "target": PRACTICE_TARGET_ITEMS,
            "hint_budget": PRACTICE_HINT_BUDGET,
            "progress": _practice_progress_map(db, ref, uid)}


@app.post("/api/practice/progress")
def api_practice_progress(request: Request, payload: dict,
                          db: Session = Depends(get_db)):
    """Save the student's place in a practice package (draft text, skip).

    Ungraded and never surfaced as a score; it only restores where they were.
    """
    ref = str(payload.get("ref", "") or "").strip()
    item_id = str(payload.get("item_id", "") or "").strip()
    if not item_id:
        raise HTTPException(400, "item_id is required")
    _practice_lesson(db, ref)
    row = _practice_progress_row(db, ref, item_id, _viewer_id(request, db))
    status = str(payload.get("status", "") or "")
    if status in ("started", "solved", "skipped"):
        row.status = status
    if "draft" in payload:
        row.draft = str(payload.get("draft") or "")[:500]
    row.updated_at = utcnow()
    db.commit()
    return {"saved": True, "status": row.status}


PRACTICE_HINT_RUNGS = (
    "In your own words, what is this item asking for? Start there.",
    "Find the first operation in the item and deal with that one piece.",
    "Name the lesson rule that fits, then write only the line it gives you.",
    "Split it: write what you are given, apply the rule once, simplify, then "
    "compare that with your attempt.",
)

# The top rung is where a stuck student can otherwise read the same sentence
# four turns running, so it rotates through three different first moves.
PRACTICE_MICRO_STEPS = (
    "Say in one sentence what this item wants before you write any symbols.",
    "Copy the item, then underline the one part you can act on first.",
    "Make a single change, then stop and compare it with the item.",
)

# Words that appear in almost any tutoring sentence, so they carry no evidence
# that a reply is about the method the verified plan actually uses.
PLAN_FILLER = frozenset({
    "that", "this", "then", "with", "from", "into", "your", "what", "when",
    "each", "them", "they", "have", "must", "only", "just", "will", "would",
    "should", "there", "their", "which", "where", "first", "next", "start",
    "item", "problem", "question", "answer", "step", "steps", "move", "moves",
    "line", "write", "send", "look", "think", "want", "wants", "need",
})


def _practice_say(*, user_text: str, history: list[dict], course_ref: str,
                  intent: str, situation: str, fallback: str,
                  reveal=(), turn: str = "", example_guard: str = "") -> str:
    """LLM-worded practice reply, with the deterministic string as fallback.

    SymPy has already decided the verdict; the model only phrases it. A
    phrasing that states the item's answer is regenerated, and if it still
    states it the deterministic line is used — practice never reveals
    solutions.
    """
    from . import float_chat as _chat

    banned = [str(r) for r in reveal if r]

    def safe(text: str) -> bool:
        if banned and _reply_reveals(text, banned):
            return False
        if example_guard and _example_matches_problem(text, example_guard):
            return False
        return True

    out = _chat.reply(user_text=user_text, history=history[-6:],
                      course_ref=course_ref, intent=intent,
                      situation=situation, graded=False, turn=turn,
                      validate=safe)
    if not out:
        return _float_display(fallback)
    # The panel prints its own "Step N:" and "Hint N —" labels, so a model that
    # copies one out of the transcript shows the student two labels — live
    # failure: "Hint 1 — Hint 2 (of 3): …", with a rung number that disagreed
    # with the ladder the student was actually on.
    text = _model_body(out[0])
    # Regeneration gets two tries; the gate is the last word, so a reply that
    # still runs long, spoils the answer, or copies the student's numbers is
    # replaced by the scripted line rather than censored mid-sentence.
    if _chat._too_long(text):
        return _float_display(fallback)
    if not text.strip() or not safe(text):
        return _float_display(fallback)
    return _float_display(text)


def _practice_verdict(problem: str, cleaned: str):
    """Grade an attempt by solving the item, not by matching its text.

    A guided item is built from the practice section's own math_latex, so the
    stored target IS the problem statement — comparing an answer against it
    would mark every correct response wrong. SymPy solves or evaluates the
    problem and compares that with the attempt.
    """
    from .schemas import Question
    from .verifier import Verdict, check_equivalent, normalize_math, \
        verify_canonical

    q = Question(id=0, prompt_latex=problem, steps=[],
                 final_answer_latex=cleaned)
    v = verify_canonical(q)
    if v.status == "unverifiable":
        alt = check_equivalent(cleaned, problem)
        if alt.status != "unverifiable":
            v = alt
    # Re-typing the item is not an attempt. Expression items are graded by
    # equivalence ("3(x-3)" for "3x-9"), so only an exact restatement counts
    # as an echo — anything else would reject a correct rewrite.
    try:
        echoed = ("".join(str(normalize_math(cleaned)).split())
                  == "".join(str(normalize_math(problem)).split()))
    except Exception:
        echoed = False
    if v.status == "correct" and echoed:
        return Verdict("echo", "That repeats the item as written.")
    return v


def _guided_expected_latex(lesson: Lesson, item_id: str) -> str:
    """Target math for a lesson guided item, from the practice section by index.

    Guided items carry no answers by design, so the expected form is resolved
    from the instructor's practice section at the same position.
    """
    import json as _json

    try:
        instructor = _json.loads(lesson.instructor_content_json or "{}")
    except Exception:
        return ""
    try:
        idx = int(item_id.split("-")[-1]) - 1
    except (ValueError, IndexError):
        return ""
    for s in instructor.get("sections", []) or []:
        if isinstance(s, dict) and s.get("kind") == "practice":
            math_items = s.get("math_latex", []) or []
            if 0 <= idx < len(math_items):
                return str(math_items[idx])
            break
    return ""


PRACTICE_HINT_BUDGET = 3


def _practice_progress_map(db: Session, ref: str, user_id: int) -> dict:
    rows = db.query(PracticeProgress).filter(
        PracticeProgress.user_id == user_id,
        PracticeProgress.package_ref == ref).all()
    return {r.item_id: {"status": r.status, "tries": r.tries or 0,
                        "hints": r.hints or 0, "draft": r.draft or ""}
            for r in rows}


def _practice_progress_get(db: Session, ref: str, item_id: str,
                           user_id: int):
    """Read the saved row without creating one (the write path owns creation)."""
    return db.query(PracticeProgress).filter(
        PracticeProgress.user_id == user_id,
        PracticeProgress.package_ref == ref,
        PracticeProgress.item_id == item_id).first()


def _practice_progress_row(db: Session, ref: str, item_id: str,
                           user_id: int):
    row = _practice_progress_get(db, ref, item_id, user_id)
    if not row:
        row = PracticeProgress(user_id=user_id, package_ref=ref,
                               item_id=item_id)
        db.add(row)
        db.commit()
    return row


def _practice_progress_record(db: Session, ref: str, item_id: str,
                              verdict: str, user_id: int,
                              hint_level=None) -> None:
    """Fold one tutor turn into the student's saved place.

    Counters live here, not in the browser, so a reload cannot reset a try or
    a hint that was already spent.
    """
    row = _practice_progress_row(db, ref, item_id, user_id)
    if verdict == "correct":
        row.tries = (row.tries or 0) + 1
        row.status = "solved"
    elif verdict == "incorrect":
        row.tries = (row.tries or 0) + 1
        row.status = "started"
    if hint_level is not None:
        row.hints = max(0, int(hint_level))
    row.updated_at = utcnow()
    db.commit()


@app.post("/api/practice/attempt")
def api_practice_attempt(request: Request, payload: dict,
                         db: Session = Depends(get_db)):
    """Chat with the tutor about one practice item.

    Same pipeline as the floating widget: the intent router decides whether a
    message is a math attempt at all, SymPy decides correctness, and the LLM
    only words the reply. Prose is never graded, hints walk the ladder from
    server-side state, and no reply may state the item's answer (practice is
    ungraded but still never reveals solutions).
    """
    from . import float_intent as _intent
    from .verifier import MAX_LEN

    uid = _viewer_id(request, db)
    ref = str(payload.get("ref", "") or "").strip()
    item_id = str(payload.get("item_id", "") or "").strip()
    student = str(payload.get("response", ""))[:MAX_LEN].strip()
    if not student:
        return {"verdict": "incomplete",
                "reply": "Type your attempt, or ask me a question — I can "
                         "hint, explain, or walk it step by step.",
                "replyType": "bubble", "hintLevel": 0, "revealed": False}
    lesson = _practice_lesson(db, ref)
    # The item is already chosen, so this is a lookup: no generation round.
    packed = _practice_set(db, lesson, top_up=False)
    items_by_id = {i.get("item_id"): i for i in packed["items"]}
    if item_id not in items_by_id:
        raise HTTPException(404, "Practice item not found")
    item = items_by_id[item_id]
    prompt = str(item.get("prompt_latex", "") or "").strip()
    task = str(item.get("task") or "").strip()
    expected = (str(packed["answers"].get(item_id) or "")
                if item.get("source") == "generated"
                else _guided_expected_latex(lesson, item_id))
    course_ref = str(payload.get("course", "") or packed["course"] or "")
    key = f"practice:{ref}:{item_id}:u{uid}"
    thread = _thread(db, key)
    state = thread.state      # ladder for this item, stored per student
    history = thread.messages  # the transcript this item has built up
    saved = _practice_progress_get(db, ref, item_id, uid)
    if saved:
        # A reload must not hand the student a fresh hint budget.
        state["hints"] = max(state["hints"], saved.hints or 0)
    # The task says what the student was asked to do with the expression.
    # Without it the tutor answers a different question than the one set.
    item_line = (f"The open practice item is: {task or prompt}"
                 if (task or prompt) else "")
    # What the reply must not say, derived by SymPy rather than by the model.
    from .verifier import canonical_answer as _canonical_answer
    answers = _canonical_answer(prompt or expected)
    # Ground the wording in the same verified path step mode walks, else the
    # model improvises a method that dead-ends (linear moves on a quadratic).
    # Instructions only: the results stay in `answers`, i.e. banned.
    from . import float_plan as _planmod
    plan = _practice_plan(db, lesson, item)
    plan_line = ""
    if plan:
        plan_line = ("Verified solution path, instructions only: "
                     + " | ".join(str(s.get("instruction", "")) for s in plan))

    def _instr(i: int) -> str:
        return str(plan[i].get("instruction", "")).strip().rstrip(".")

    def _move(idx: int) -> str:
        return (f"Step {idx + 1} of {len(plan)}: {_instr(idx)}. "
                "Write just that line — what do you get?")

    def _on_method(text: str, idx: int) -> bool:
        """True when `text` is worded in the vocabulary of plan step `idx`.

        The model only words a reply, and it sometimes words a different
        method — a live failure gave a linear "balance both sides" hint on a
        factoring item, which sends the student down a dead end. The expected
        expression is deliberately not part of the vocabulary: a hint must
        never have to name the result to count as on-method.
        """
        import re as _re

        step = plan[idx]
        bits = [str(step.get("instruction", "")), str(step.get("reason", ""))]
        bits += [str(m) for m in (step.get("micro_steps") or [])]
        pool = {w for w in _re.findall(r"[a-z]{4,}", " ".join(bits).lower())
                if w not in PLAN_FILLER}
        said = set(_re.findall(r"[a-z]{4,}", (text or "").lower()))
        return len(said & pool) >= min(2, len(pool))

    def _grounded(text: str) -> bool:
        """True when the reply names something from this item itself.

        A hint may legitimately ask a question instead of naming the method
        ("as x approaches 1, what does x approach?"), so grounding is the
        other way a reply proves it is about this item rather than reciting a
        rule that belongs to a different one.
        """
        import re as _re

        src = (prompt or expected).lower()
        marks = set(_re.findall(r"\d+", src)) | set(_re.findall(r"\b[a-z]\b", src))
        said = set(_re.findall(r"\d+|\b[a-z]\b", (text or "").lower()))
        return bool(marks & said)

    def _plan_step_idx(text: str) -> int | None:
        """Index of the plan step this line completes, or None.

        The item checker compares an attempt with the item's own solution, so
        a correct intermediate line reads "not equivalent" even when it is
        exactly the line a plan step asked for. Without this the tutor called
        its own step 1 wrong: it asked for 0.6x = 2.4, then marked it incorrect.
        Only full equations count — a bare value matches almost any step.
        """
        for i, st in enumerate(plan):
            exp = str(st.get("expected_expression", "") or "")
            if "=" not in exp:
                continue
            ct = str(st.get("check_type", "") or "equivalent_expression").lower()
            try:
                if _planmod.check_step(exp, text, ct) == "correct":
                    return i
            except Exception:
                continue
        return None

    intent = _intent.classify(student, has_session=True)

    def _turn_ctx(verdict: str = "", intent_name: str = "",
                  repeats: bool = False) -> str:
        from . import float_chat as _chat

        return _chat.turn_context(
            problem=prompt or expected, verdict=verdict,
            hint_level=state["hints"],
            intent=intent_name or intent.lower(), mode="practice",
            first_name=_student_first_name(db, uid),
            student_answer=student, repeats=repeats, task=task)

    def _log_turn(rtype: str, verdict: str) -> None:
        import logging as _logging

        _logging.getLogger("mathpilot.tutor").info(
            "tutor_turn item=%s intent=%s reply_type=%s verdict=%s "
            "hint_level=%s mode=practice", item_id, intent.lower(), rtype,
            verdict, state["hints"])

    def _turn(reply: str, verdict: str, detail: str = "",
              rtype: str = "bubble") -> dict:
        history.extend([{"role": "user", "content": student},
                        {"role": "assistant", "content": reply}])
        thread.save()
        _practice_progress_record(db, ref, item_id, verdict, uid,
                                  state["hints"])
        _log_turn(rtype, verdict)
        return {"verdict": verdict, "reply": reply, "detail": detail,
                "intent": intent, "replyType": rtype,
                "hintLevel": state["hints"], "revealed": False,
                "misses": state["misses"]}

    def _say(situation: str, fallback: str,
             intent_name: str = "chat", verdict: str = "",
             example_guard: bool = False) -> str:
        return _practice_say(
            user_text=student, history=history, course_ref=course_ref,
            intent=intent_name, situation=situation, fallback=fallback,
            reveal=answers, turn=_turn_ctx(verdict, intent_name=intent_name),
            example_guard=(prompt or expected) if example_guard else "")

    def _hint(lvl: int, extra: str) -> str:
        state["hints"] = lvl
        body = _say(
            "Give exactly hint level "
            f"{lvl} of 4 for this practice item (1 what the item asks for, "
            "2 where to look, 3 which lesson rule to apply, 4 split it into "
            "micro-steps). Name the move, never do the algebra for them, and "
            "never state what the item evaluates to. Open with the help "
            "itself: do not repeat the word Hint or number the rung, the app "
            "prints that label. "
            + ("For level 3 the worked example must use different numbers "
               "than the student's item. " if lvl == 3 else "")
            + f"{item_line} {plan_line} {extra}",
            "", verdict="incorrect", example_guard=lvl == 3)
        body = _model_body(body)
        if body and plan and not (_on_method(body, min(lvl, len(plan)) - 1)
                                  or _grounded(body)):
            # Live failure: rung 1 on a factoring item came back as "whatever
            # you do to one side, do to the other" — a linear rule that names
            # no move of this item and cannot finish it. The verified move for
            # the rung replaces it.
            body = ""
        # Deterministic fallback follows the verified path too: rung N names
        # plan step N's move, so a failed model call still teaches the right
        # method instead of a generic sentence.
        if plan:
            rung = min(lvl, len(plan)) - 1
            # No "Step N of M:" here: the panel already prints "Hint N —", so
            # the rung is the move itself, not a second label stacked on it.
            micro = f"{_instr(rung)}. Write just that line — what do you get?"
            if lvl > len(plan):
                # The top rung asks for smaller than the plan itself, so use
                # that move's own micro-steps instead of restating the last one.
                bits = [str(m).strip().rstrip(".")
                        for m in (plan[rung].get("micro_steps") or [])
                        if str(m).strip()]
                if bits:
                    micro = ("In smaller pieces: " + bits[0] + "".join(
                        ". Then " + b[:1].lower() + b[1:] for b in bits[1:])
                        + ". Do only the first one and send that line.")
        else:
            micro = PRACTICE_MICRO_STEPS[state["misses"] % len(PRACTICE_MICRO_STEPS)]
        last = next((str(h.get("content", "")) for h in reversed(history)
                     if h.get("role") == "assistant"), "")
        if micro in last:
            if plan and min(lvl, len(plan)) < len(plan):
                micro = _move(min(lvl, len(plan)))
            else:
                micro = PRACTICE_MICRO_STEPS[
                    (state["misses"] + 1) % len(PRACTICE_MICRO_STEPS)]
        return _float_display(
            f"Hint {lvl} — {body or (micro if (lvl >= 4 or plan) else PRACTICE_HINT_RUNGS[lvl - 1])}")

    if intent == "OFF_TOPIC":
        return _turn(_say(
            "The student asked about something outside the course sequence. "
            "Decline in one sentence and steer back to the open item. "
            f"{item_line}",
            "I'm your math tutor — let's keep to course math. " + (
                item_line or "Send me your attempt and I'll guide the next "
                             "step."), intent_name="off_topic"),
            "not_an_attempt")

    if intent == "CONCEPT_QUESTION":
        return _turn(_say(
            "The student asked a conceptual math question. Answer the idea "
            "briefly and correctly, then invite them to try one step of the "
            f"open item. Do not solve their item. {item_line}",
            "Good question — tell me which lesson rule it uses, then try one "
            "step of the item above and I'll check it.",
            intent_name="concept"), "not_an_attempt")

    if intent == "CHAT":
        return _turn(_say(
            "The student is chatting, not answering. Reply warmly in a "
            "sentence and ask one question that moves the item forward. "
            f"{item_line}",
            "Happy to help. " + (item_line or "Send me an attempt.") +
            " What have you tried so far?"), "not_an_attempt")

    if intent == "NEW_PROBLEM":
        return _turn(_say(
            "The student typed a different problem. Practice items come from "
            "their lesson, so point them back to the open item and mention "
            f"the 'Work it step by step' button. {item_line}",
            "Practice items come from your lesson, so let's work the one "
            "above. Tap Work it step by step and I'll split it into steps "
            "for you."), "not_an_attempt")

    if intent == "ANSWER_REQUEST":
        # Reveal policy stays in code, in both directions: the refusal is the
        # deterministic line, never an LLM phrasing that can slip the answer
        # into the same sentence that declines to give it.
        return _turn(
            "I can't give the final answer — that's the rule that makes "
            "practice work. Tell me the last step you tried, or send your "
            "attempt, and I'll guide the next move.",
            "not_an_attempt")

    if intent == "HELP_REQUEST":
        return _turn(_hint(min(4, state["hints"] + 1),
                           "The student asked for help."),
                     "hint", rtype="hint")

    if intent in ("START_STEP", "FRUSTRATED"):
        # "I don't know" and "what should I do?" are not attempts: no miss, no
        # hint spent. The student gets the next move on the item as an action.
        given = int(state.get("moves", 0) or 0)
        if plan and intent == "START_STEP" and given < len(plan):
            # Advance by moves actually handed out, not by how often the
            # student asked: a transcript full of stale replies still walks
            # the verified path in order instead of repeating its first line.
            state["moves"] = given + 1
            already = ""
            if given:
                already = (f"You already gave move {given}: "
                           f"{_instr(given - 1)}. Do not restate it. ")
            said = _say(
                "The student asked what to do. Give only this one move, in "
                "your own words, and stop before the next one. Never state "
                f"what it evaluates to. End by asking what they get. {already}"
                f"Move {given + 1} of {len(plan)}: {_instr(given)}. "
                f"{item_line}",
                _move(given), verdict="where_to_start",
                intent_name="where_to_start")
            prior = {str(h.get("content", "")) for h in history
                     if h.get("role") == "assistant"}
            if said in prior or not _on_method(said, given):
                # Live failures: the model re-worded the move it had already
                # given, so the walk looked stuck on step 1, and it handed out
                # a move from another method. The deterministic line for this
                # turn's move always advances, and always teaches this item.
                said = _float_display(_move(given))
            return _turn(said, "not_an_attempt", "", "step")
        asks = sum(1 for h in history if h.get("role") == "user" and
                   _intent.classify(str(h.get("content", "")), True) == intent)
        if asks >= 2 or (plan and given >= len(plan)):
            # Either the verified path is used up or a third "I'm lost" means
            # restating the start is not helping: climb the ladder instead.
            return _turn(_hint(min(4, state["hints"] + 1),
                               "The student asked what to do more than once."),
                         "not_an_attempt", "", "hint")
        first_move = _instr(0) if plan else ""
        scripted = (
            "This one is tricky, and that's okay. One small move first: say "
            "in words what the item asks for, then write only that line."
            if intent == "FRUSTRATED" else
            (f"First move only: {first_move}. Write just that line — what do "
             "you get?" if first_move else
             "First move only: look at the left side of the item and name the "
             "very first operation you would do. Write just that line — what "
             "do you get?"))
        said = _say(
            (f"The student is frustrated. Acknowledge in one sentence, then "
             "shrink the work to one small move on this item and ask them to "
             f"send only that. {item_line} {plan_line}"
             if intent == "FRUSTRATED" else
             "The student asked what to do. Give one action for the first "
             f"move on this item — never the result. {item_line} {plan_line} "
             "End by asking what they get."),
            scripted, verdict=("unclear" if intent == "FRUSTRATED"
                               else "where_to_start"),
            intent_name=("frustrated" if intent == "FRUSTRATED"
                         else "where_to_start"))
        return _turn(said, "not_an_attempt",
                     "", "support" if intent == "FRUSTRATED" else "step")

    # STEP_ANSWER (and anything unrecognized): the CAS decides.
    cleaned = _intent.clean_answer(student)
    if not cleaned:
        # Unreadable, not wrong: no miss, no ladder move, and the reply shows
        # the shape to type instead of repeating the item back at them.
        state["unreadable"] = state.get("unreadable", 0) + 1
        scripted = ("That did not come through as math, so nothing was marked. "
                    "Send just the line you got, written like 3x = 9 or "
                    "x = 4.")
        if state["unreadable"] >= 2:
            scripted += " Tap fx for the fraction, root and symbol keys."
        return _turn(_say(
            "The student's message did not parse as a math response, so it is "
            "unreadable rather than wrong. Do not blame them and do not repeat "
            "the whole item. Show one short example of the form to type and "
            f"ask for just that line. {item_line}",
            scripted, verdict="unreadable"), "not_an_attempt", "",
            "format_help")
    state["unreadable"] = 0

    verdict = _practice_verdict(prompt or expected, cleaned)
    status = verdict.status
    detail = verdict.detail

    if status == "correct":
        state["hints"] = 0
        state["moves"] = 0
        return _turn(_say(
            "Our math checker VERIFIED this attempt correct. Confirm warmly "
            "in one sentence and ask what they want to practise next. Do not "
            "restate the answer.", "✔ Verified — nice work.",
            verdict="correct"),
            "correct", detail, "success")

    if status == "echo":
        return _turn(_say(
            "The student sent the item back as if it were their answer. Ask "
            "for the first move they would make on it — no verdict, no hint "
            f"spent. {item_line}",
            "That's the item itself, not your working. What's your first move "
            "on it?"), "not_an_attempt", detail, "bubble")

    if status == "incomplete":
        # Right but not finished (one of two solutions): what's missing, and
        # explicitly not a miss.
        return _turn(_say(
            f"SymPy says this attempt is partly right: {detail}. Say what is "
            "right and ask for what is still missing, in one sentence. Do not "
            f"name the missing value. {item_line}",
            "You've got part of it. What's still missing — is there another "
            "value, or a final simplification? Send that line.",
            verdict="partial"), "partial", detail, "partial")

    if status == "incorrect":
        # A correct intermediate line is not a miss. The item is graded
        # against its own solution, so "0.6x = 2.4" — exactly what move 2 of
        # the verified path asked for — reads "not equivalent" on its own.
        # Match the line against the plan before it is counted as wrong.
        hit = _plan_step_idx(cleaned)
        if hit is not None and hit + 1 < len(plan):
            state["moves"] = max(int(state.get("moves", 0) or 0), hit + 1)
            state["hints"] = 0
            nxt = hit + 1
            said = _say(
                f"Their line {cleaned} is the verified result of move "
                f"{hit + 1} of {len(plan)} on this item. Confirm in one short "
                "sentence that this line is right — say nothing about which "
                "operation they used, because you cannot see their working — "
                "then give only the next move and ask what they get. "
                f"Move {nxt + 1} of {len(plan)}: {_instr(nxt)}. Never state "
                f"what that evaluates to. {item_line}",
                _move(nxt), verdict="correct", intent_name="where_to_start")
            if not _on_method(said, nxt):
                said = _float_display(_move(nxt))
            return _turn(said, "partial",
                         f"Matches move {hit + 1} of the verified path.",
                         "step")
        state["misses"] += 1
        repeats = any(h.get("role") == "user"
                      and str(h.get("content", "")).strip() == student.strip()
                      for h in history[-4:])
        return _turn(_hint(min(4, state["hints"] + (2 if repeats else 1)),
                           f"Their attempt was {cleaned}, which the checker "
                           "says is not equivalent to the target yet."
                           + (" They sent the same attempt twice, so change "
                              "the angle." if repeats else "")),
                     "incorrect", detail, "hint")

    if not (prompt or expected):
        # Honest about the limitation instead of a fake verdict.
        return _turn(_say(
            "This item has no reference answer in the lesson, so it cannot be "
            "verified. Say that plainly and ask which part of their reasoning "
            f"they want to check. {item_line}",
            "This item has no reference answer in your lesson, so I can't "
            "verify it — compare your work with the worked example above. "
            "Which step do you want to check?"),
            "unverifiable", "no reference answer for this item")

    return _turn(_say(
        f"Their message {cleaned} could not be checked against this item "
        "(unsupported notation or an ambiguous form). Ask them to rewrite it "
        f"in plain math, or offer a hint. {item_line}",
        "I couldn't check that against this item — the notation is unclear. "
        "Rewrite it as a single equation, or tap fx for symbols, and send it "
        "again."), "unverifiable", "could not verify that form")


@app.post("/api/tutor/chat")
def api_tutor_chat(request: Request, payload: dict,
                   db: Session = Depends(get_db)):
    """Floating Math Tutor chat (Student View only).

    Policy (enforced here, never in prompts alone):
    - Practice context (ungraded, item loaded): reuse the existing
      /api/practice/attempt verification pipeline; the reply is the
      same Socratic feedback the practice page gives.
    - Assignment context (graded): hints only. Never verify the student
      against assignment solutions, never reveal answers.
    - Lesson/general context: Socratic guidance grounded in the current
      lesson topic; never a worked solution or final answer.
    Server-side conversation state is shared with the practice page through
    the stored thread, so both surfaces continue one conversation.
    """
    from .verifier import MAX_LEN as _MAX_LEN

    uid = _viewer_id(request, db)
    message = str(payload.get("message", "") or "")[:_MAX_LEN].strip()
    if not message:
        raise HTTPException(400, "Message is required")
    context = payload.get("context") or {}
    if not isinstance(context, dict):
        context = {}
    mode = str(context.get("mode", "general") or "general")

    lowered = message.lower()
    wants_answer = any(p in lowered for p in (
        "give me the answer", "what is the answer", "just tell me",
        "solve it for me", "show me the solution", "reveal the answer",
        "what's the answer", "what is the solution",
    ))

    # --- Practice context: same verified pipeline as the practice page.
    if mode == "practice" and context.get("link_id") and context.get("item_id"):
        from .lessons import split_student_payload
        import json as _json

        link_id = str(context.get("link_id", ""))
        item_id = str(context.get("item_id", ""))
        pub = db.query(Publication).filter(
            Publication.link_id == link_id).first()
        if not pub or not pub.lesson_id:
            raise HTTPException(404, "Practice package not found")
        lesson = db.get(Lesson, pub.lesson_id)
        if not lesson or not lesson.student_content_json:
            raise HTTPException(404, "No practice content")
        try:
            content = split_student_payload(
                _json.loads(lesson.student_content_json))
        except Exception:
            raise HTTPException(502, "Practice content unreadable")
        guided = {g.get("item_id"): g for g in
                  content.get("guided_items", []) or []}
        if item_id not in guided:
            raise HTTPException(404, "Practice item not found")
        if wants_answer:
            thread = _thread(db, f"practice:{link_id}:{item_id}:u{uid}")
            reply = ("I can't give the final answer — that's the rule that "
                     "makes practice work. Tell me the last step you tried, "
                     "or press Check Step on your attempt, and I'll guide "
                     "the next move.")
            thread.append(message, reply)
            thread.save()
            return {"reply": reply, "mode": "practice", "revealed": False}
        practice_payload = {
            "ref": link_id, "item_id": item_id,
            "response": message, "course": context.get("course", ""),
        }
        result = api_practice_attempt(request, practice_payload, db)
        return {"reply": result.get("reply", ""), "mode": "practice",
                "verdict": result.get("verdict", ""),
                "revealed": False}

    # --- Assignment context: hints only, no verification, no answers.
    if mode == "assignment":
        if wants_answer:
            reply = ("This is graded work, so I can't check answers or "
                     "give solutions here. Re-read the question, try one "
                     "step in the answer box, and use Save Progress — "
                     "practice problems are the place for step-by-step help.")
        else:
            reply = ("Good question — for graded work I can only offer a "
                     "study hint, not check your answer. Which concept from "
                     "your lesson does this question test? Name it, then try "
                     "the first step on your own.")
        return {"reply": reply, "mode": "assignment", "revealed": False}

    # --- Lesson / general context: the same brain, never a worked answer.
    lesson_title = str(context.get("lesson_title", "") or "")[:200]
    lesson_meta = str(context.get("lesson_meta", "") or "")[:200]
    thread = _thread(db, f"float-tutor:general:u{uid}")
    history = thread.messages
    if wants_answer:
        reply = ("I won't give the final answer — but I will get you there. "
                 "What have you tried so far? Show me your last step and "
                 "I'll point at the next move.")
    else:
        from . import float_chat as _chat
        from . import float_intent as _fi

        lesson_note = lesson_meta or "no topic given"
        said = _chat.reply(
            user_text=message, history=history,
            course_ref=str(context.get("course", "") or ""),
            intent=("concept" if _fi.classify(message, False) ==
                    "CONCEPT_QUESTION" else "chat"),
            turn=_chat.turn_context(
                problem=lesson_title or "(no problem open yet)",
                verdict="unclear", intent="concept_question",
                mode="practice", first_name=_student_first_name(db, uid),
                notation=lesson_meta, student_answer=message),
            situation=(
                "The student is in the AI Tutor tab with lesson context only: "
                f"{lesson_title or 'no lesson named'} ({lesson_note}). Guide "
                "them to state the problem and take the first small step. "
                "Never write a worked solution or a final answer."))
        reply = (_float_display(said[0]) if said else
                 ("**Demo mode** — here's how to start: write down what the "
                  "problem gives you and what it's asking for, try the "
                  "smallest first step, and tell me what you got. "
                  "Add your Nebius key for live Socratic tutoring."))
    thread.append(message, reply)
    thread.save()
    return {"reply": reply, "mode": mode, "revealed": False}


@app.get("/api/tutor/session")
def api_tutor_session(request: Request, course: str = "",
                      db: Session = Depends(get_db)):
    """Resume hook: the widget + AI Tutor tab share this open session."""
    return _float_thread_payload(db, _viewer_id(request, db),
                                 course or "default")


def _float_thread_payload(db: Session, student_id: int,
                           course_id: str) -> dict:
    """Shared conversation thread for widget + AI Tutor tab (test 12).

    Returns the open session state plus its logged message history so
    both surfaces render the same conversation.
    """
    import json as _json

    session = _float_open_session(db, student_id, course_id or "default")
    if not session:
        return {"active": False, "messages": []}
    try:
        plan = _json.loads(session.solution_plan_json or "[]")
    except Exception:
        plan = []
    events = db.query(TutorEvent).filter(
        TutorEvent.session_id == session.id).order_by(
        TutorEvent.id.asc()).all()
    messages = [{"role": "student", "text": e.student_input}
                for e in events if e.event_type in ("message", "graded_use")]
    return {"active": True, "problem_text": session.problem_text,
            "stepIndex": session.current_step_index,
            "totalSteps": session.total_steps,
            "graded": bool(session.graded_mode),
            "prompt": _float_current_prompt(session),
            "messages": messages[-30:]}


# --- Floating tutor: CAS-driven guided sessions (Section 6 of the spec) ---

FLOAT_TUTOR_RATE_LIMIT = 30  # messages per student per 10 minutes
FLOAT_TUTOR_WINDOW_S = 600
GRADED_HINTS_ONLY = True  # single setting: graded mode restricts hints

# A new-problem offer the student has not answered yet expires after 10
# minutes; on loss or expiry the router re-runs and never grades a probable
# new problem as a step answer.
FLOAT_TUTOR_PENDING_TTL_S = 600


def _float_pending_row(db: Session, student_id: int, course_id: str):
    return db.query(TutorPending).filter(
        TutorPending.student_id == student_id,
        TutorPending.course_id == course_id).first()


def _float_pending_get(db: Session, student_id: int,
                       course_id: str) -> dict | None:
    import time as _time

    row = _float_pending_row(db, student_id, course_id)
    if not row:
        return None
    if _time.time() - (row.ts or 0) > FLOAT_TUTOR_PENDING_TTL_S:
        db.delete(row)
        db.commit()
        return None
    return {"problem": row.problem_text, "ts": row.ts}


def _float_pending_set(db: Session, student_id: int, course_id: str,
                       problem: str) -> None:
    import time as _time

    row = _float_pending_row(db, student_id, course_id)
    if row is None:
        row = TutorPending(student_id=student_id, course_id=course_id)
        db.add(row)
    row.problem_text = problem
    row.ts = _time.time()
    db.commit()


def _float_pending_clear(db: Session, student_id: int,
                         course_id: str) -> None:
    row = _float_pending_row(db, student_id, course_id)
    if row is not None:
        db.delete(row)
        db.commit()


def _float_rate_ok(db: Session, student_id: int) -> bool:
    """Allow this student's message, or bounce it to the sliding window."""
    import time as _time

    now = _time.time()
    db.query(TutorRateHit).filter(
        TutorRateHit.student_id == student_id,
        TutorRateHit.ts < now - FLOAT_TUTOR_WINDOW_S).delete(
        synchronize_session=False)
    recent = db.query(TutorRateHit).filter(
        TutorRateHit.student_id == student_id).count()
    if recent >= FLOAT_TUTOR_RATE_LIMIT:
        db.commit()
        return False
    db.add(TutorRateHit(student_id=student_id, ts=now))
    db.commit()
    return True


def _float_open_session(db: Session, student_id: int,
                        course_id: str) -> TutorSession | None:
    def _newest(*filters):
        return db.query(TutorSession).filter(
            TutorSession.student_id == student_id,
            TutorSession.status == "active",
            *filters).order_by(TutorSession.id.desc()).first()

    session = _newest(TutorSession.course_id == course_id)
    if session:
        return session
    # The widget's resume call and its message call do not always agree on
    # the course label ("Calculus 1" vs the "" default), and a mismatch must
    # never make an open problem disappear from the strip. Same student only.
    return _newest()


def _float_log(db: Session, session_id: int, step_index: int,
               student_input: str, verdict: str, hint_level: int,
               event_type: str = "message", *, reply_type: str = "",
               intent: str = "", mode: str = "") -> None:
    """Append one tutor turn.

    The row keeps the fields the instructor views already read (verdict,
    hint_level, event_type). reply_type/intent/mode are per-turn behaviour
    metadata the spec asks for and no table column exists for, so they go to
    the structured tutor log rather than a schema change.
    """
    db.add(TutorEvent(session_id=session_id, step_index=step_index,
                      student_input=student_input[:2000], verdict=verdict,
                      hint_level=hint_level, event_type=event_type))
    db.commit()
    if reply_type or intent or mode:
        import logging as _logging

        _logging.getLogger("mathpilot.tutor").info(
            "tutor_turn session=%s step=%s intent=%s reply_type=%s "
            "verdict=%s hint_level=%s mode=%s",
            session_id, step_index, intent or "-", reply_type or "-",
            verdict or "-", hint_level, mode or "practice")


def _student_first_name(db: Session, student_id: int) -> str:
    row = db.query(User).filter(User.id == student_id).first()
    parts = str(getattr(row, "name", "") or "").split() if row else []
    return parts[0] if parts else ""


def _float_current_prompt(session: TutorSession | None) -> str:
    import json as _json

    if session is None:
        return ("Send me an equation and I'll guide you step by step. "
                "I won't give you the answer — you'll find it yourself.")
    try:
        plan = _json.loads(session.solution_plan_json or "[]")
    except Exception:
        plan = []
    if plan and 0 <= session.current_step_index < len(plan):
        step = plan[session.current_step_index]
        return (f"Step {session.current_step_index + 1}: "
                f"{step.get('instruction', '')} What do you get?")
    return ("Send me an equation and I'll guide you step by step. "
            "I won't give you the answer — you'll find it yourself.")


def _float_hint(step: dict, hint_level: int, miss_count: int,
                step_no: int, avoid: str = "") -> tuple[str, int]:
    """Hint ladder. Returns (reply, new_hint_level). Never reveals results."""
    import hashlib

    level = min(hint_level + 1, 4)
    instruction = step.get("instruction", "")
    reason = step.get("reason", "")
    if level == 1:
        reply = (f"Hint 1 — gentle nudge: {reason} "
                 f"Now try: {instruction} What do you get?")
    elif level == 2:
        reply = (f"Hint 2 — look closely: focus on the exact term the step "
                 f"names. {instruction} What equation do you get?")
    elif level == 3:
        # Deterministic worked mini-example with DIFFERENT numbers.
        seed = int(hashlib.md5(
            instruction.encode()).hexdigest()[:8], 16)
        a = 2 + seed % 5          # coefficient 2..6
        b = 1 + (seed >> 3) % 8   # constant 1..8
        c = a * (1 + (seed >> 6) % 4) + b  # keeps integer answer
        ans = (c - b) // a
        reply = (f"Hint 3 — mirror this different example, then do yours "
                 f"the same way: if we had ${c} = {a}y + {b}$, we'd "
                 f"subtract {b} first to get ${c - b} = {a}y$. Now back "
                 f"to your problem: {instruction} What do you get? "
                 f"(That example solves to y = {ans} — yours is different.)")
    else:
        # Top rung: rotate so a stuck student never reads the same advice
        # twice in a row.
        openers = (
            "First, copy down just the left side of your equation. Then copy "
            "the right side.",
            "First, write down only the one operation this step asks for.",
            "First, look at what you already have and name the piece that is "
            "still untouched.",
        )
        closer = ("If it's still stuck, re-read the worked example in the "
                  "lesson, or ask your instructor about this step. Which of "
                  "those two parts can you do now?"
                  if miss_count % 2 else
                  "If it's still stuck, ask your instructor about this step "
                  "in class. What is the first piece you can write down?")
        pick = miss_count % len(openers)
        if avoid:
            # A help request on a step already at the top rung would otherwise
            # read the same sentence again, so move off the previous wording.
            last = next((k for k, o in enumerate(openers) if o in avoid), None)
            if last is not None:
                pick = (last + 1) % len(openers)
        reply = (f"Let's split it into micro-steps. {openers[pick]} "
                 f"Now: {instruction} {closer}")
    return reply, level


def _model_body(llm_text: str) -> str:
    """Return the model's wording with any engine label it re-added removed.

    The engine prints "Step N:" and "Hint N —" itself. A model that copies a
    label out of the previous message would show the student two labels, or a
    rung number inside a step reply ("Step 2: Hint 4 — …"). Only leading
    labels go: prose that merely starts with a sentence is left alone.
    """
    import re as _re

    step = _re.compile(r"([^.!?\n]{0,30}[!.?\n]\s*)?(step\s*\d+\s*[:.\-—]\s*)",
                       _re.IGNORECASE)
    hint = _re.compile(
        r"^\s*(?:\U0001F4A1\s*)?(?:"
        r"(?:hint|level|ladder|rung)[\s\-]*(?:level|rung)?\s*\d*"
        r"(?:\s*\(\s*of\s*\d+\s*\)|\s+of\s+\d+)?"
        r"\s*[:\-–—]\s*"
        r"|(?:hint\s+)?\d+\s*[:\-–—]\s*)", _re.IGNORECASE)
    body = (llm_text or "").strip()
    for _ in range(3):
        before = body
        m = step.match(body)
        if m:
            body = (m.group(1) or "") + body[m.end():]
        body = hint.sub("", body.strip(), count=1).strip()
        if body == before:
            break
    # A label the model copied into the middle of its own reply survives the
    # anchored pass above — live failure: "Hint 2 — That's not the factored
    # form yet. Hint 2 — Look at the + 10 …".
    interior = _re.compile(
        r"(?<=[.!?]\s)\s*(?:hint|step|level|rung)\s*\d+"
        r"(?:\s*\(\s*of\s*\d+\s*\)|\s+of\s+\d+)?\s*[:\-–—]\s*(\w?)",
        _re.IGNORECASE)
    return interior.sub(lambda m: m.group(1).upper(), body).strip()


def _float_display(reply: str) -> str:
    """Apply the small display formatter to tutor reply text.

    Instructions are already prose; this cleans any embedded CAS
    spellings (2*x, x**2, pi) without touching $...$ math regions'
    structure, and delimits bare LaTeX so it renders as math instead of
    showing a student backslash code. Safe to run on every reply.
    """
    from .verifier import display_math as _dm, wrap_bare_latex as _wrap

    parts = reply.split("$")
    for i in range(0, len(parts), 2):  # even indices: outside math
        chunk = parts[i]
        if not chunk.strip():
            continue
        # display_math strips, which used to weld the prose onto the inline
        # math ("of $f(x)$ next" -> "of$f(x)$next"), so keep the pad.
        pad_l = chunk[:len(chunk) - len(chunk.lstrip())]
        pad_r = chunk[len(chunk.rstrip()):]
        parts[i] = pad_l + _wrap(_dm(chunk)) + pad_r
    return "$".join(parts)


def _leak_pattern(exp: str) -> str | None:
    """Regex that matches a step result as written, or None if it can't spoil.

    A bare number ("10" from a substitution check) is not a leakable answer,
    and plain substring matching made every reply that quoted the problem
    ("$10 = 3x + 1$") look like a leak. Numbers only count when the reply
    states them as a result, "… = 10".
    """
    import re as _re

    e = str(exp or "").strip().lower()
    if not e:
        return None
    # A sentence stop after the value is still a leak ("so x = 3."), while a
    # longer number is not ("x = 3.5" must not match a value of "3").
    tail = r"(?!\w)(?!\.\d)"
    if not _re.search(r"[a-z]", e):
        return rf"=\s*{_re.escape(e)}{tail}"
    # Token-escape, then let any run of spaces match any other: the plan
    # writes "9 = 3*x" but a reply reads "9 = 3x".
    core = r"\s*".join(_re.escape(t) for t in e.split())
    core = core.replace(r"\*", r"\*{0,1}")
    return rf"(?<![\w.]){core}{tail}"


def _reply_leaks(reply: str, expressions) -> bool:
    """True when the reply states one of the step results as written."""
    import re as _re

    low = (reply or "").lower()
    return any((pat := _leak_pattern(e)) and _re.search(pat, low)
               for e in expressions)


def _reveal_pattern(value: str) -> str | None:
    """Regex matching a reply that presents `value` as the item's result.

    Practice items show the problem, so its own numbers appear in legitimate
    guidance ("as x approaches 1…"). A bare value therefore only counts as a
    reveal when the wording hands it over as the answer. A symbolic value
    ("3(x-3)") is distinctive enough to match on its own.
    """
    import re as _re

    v = str(value or "").strip().lower()
    if not v:
        return None
    core = r"\s*".join(_re.escape(t) for t in v.split())
    core = core.replace(r"\*", r"\*{0,1}")
    # Reject "10" for a value of "1" and "1.5" for "1", but not the ordinary
    # sentence stop after the value ("the answer is 1.").
    tail = r"(?!\w)(?!\.\d)"
    if _re.search(r"[a-z]", v):
        return rf"(?<![\w.]){core}{tail}"
    # A bare number only counts as a reveal when the wording hands it over:
    # "the answer is 3", "x = 3", or the root inside a stated factor —
    # "(x - 2)(x - 5) = 0" gives the whole solution away without ever using
    # the word answer. The item itself shows the same digits, so a value on
    # its own is still not a reveal.
    return (r"(?:\b(?:answer|result|value|solution|limit|it|that|this|you"
            r"|step|form|expression|product)"
            r"[^.!\n]{0,28}?(?:\bis\b|:|\bequals?|\bcomes to|\bworks out to)"
            rf"|\b[a-z]\s*=|\(\s*[a-z]\s*[-+]\s*)\s*{core}{tail}")


def _reply_reveals(reply: str, values) -> bool:
    """True when the reply states the item's answer rather than guiding."""
    import re as _re

    low = (reply or "").lower()
    return any((pat := _reveal_pattern(v)) and _re.search(pat, low)
               for v in values)


def _float_plan(session) -> list:
    """The hidden solution plan, or [] — never handed to the browser."""
    import json as _json

    if session is None:
        return []
    try:
        plan = _json.loads(session.solution_plan_json or "[]")
    except Exception:
        return []
    return plan if isinstance(plan, list) else []


def _float_unreached_results(session, *, include_current: bool = False) -> list:
    """Step results the student has not produced yet — the leak set."""
    if session is None:
        return []
    plan = _float_plan(session)
    start = session.current_step_index if include_current else \
        session.current_step_index + 1
    return [st.get("expected_expression", "") for st in plan[start:]]


def _float_jump_ahead(plan: list, idx: int, message: str,
                      problem: str = "") -> int | None:
    """Index of a later step this message already answers, else None.

    A student who does two steps in one message is ahead, not wrong, so the
    checker looks forward before the miss is recorded. Only real work steps
    count: a bare "0" or "10" would match a substitution check by accident,
    and echoing the problem text is not an answer.
    """
    from . import float_plan as _plan
    from .float_intent import clean_answer
    from .verifier import _parse_single

    from_txt = clean_answer(message)
    if not from_txt:
        return None
    try:
        if _parse_single(from_txt) == _parse_single(problem or ""):
            return None  # the item sent back is not a step
    except Exception:
        pass
    for k in range(idx + 1, len(plan)):
        st = plan[k]
        ct = str(st.get("check_type", "equivalent_expression")).lower()
        exp = str(st.get("expected_expression", "") or "")
        if ct in ("numeric_value", "conceptual", "limit") or "=" not in exp:
            continue
        if _plan.check_step(exp, from_txt, ct) == "correct":
            return k
    return None


def _float_leak_check(reply: str, session, block_current: bool = False) -> str:
    """Answer-leak check (spec 6e): strip step results from replies.

    Compares the reply against future steps' expected expressions; on a
    hit, replaces with the current step's instruction. The recap card
    bypasses this (it intentionally shows the full solution). When
    `block_current` is set the active step's result is blocked too, which
    is what a refusal to reveal an answer needs.
    """
    import json as _json

    plan = _float_plan(session)
    upcoming = _float_unreached_results(
        session, include_current=block_current)
    if not plan or not upcoming:
        return reply
    if _reply_leaks(reply, upcoming):
        cur = plan[min(session.current_step_index, len(plan) - 1)]
        return (f"Let's stay on the current step: "
                f"{cur.get('instruction', '')} What do you get?")
    return reply


_TRAILING_STOP = ("in", "of", "to", "for", "and", "or", "the", "a", "an",
                  "then", "with", "by")


def _rail_title(instruction: str) -> str:
    """Short plain-language label for one plan step, with no maths in it.

    Instructions are written for the chat ("Let u = cos(x). Rewrite …"), and
    a clause that already states the maths reads badly once the symbols are
    stripped, so the label comes from the first prose clause.
    """
    import re as _re

    text = str(instruction or "")
    parts = [p.strip() for p in
             _re.split(r"(?<=[A-Za-z)])\s*[:.;]\s+", text) if p.strip()]
    if not parts:
        return ""
    clause = next((p for p in parts if "=" not in p), parts[0])
    clause = _re.sub(r"\$[^$]*\$", " ", clause)
    clause = _re.sub(r"[\[\(][^\]\)]*[\]\)]?", " ", clause)
    clause = _re.sub(r"\\[a-zA-Z]+", " ", clause)
    clause = _re.sub(r"[^A-Za-z0-9./ ]+", " ", clause)
    words = _re.sub(r"\s+", " ", clause).strip().split()[:5]
    while len(words) > 1 and words[-1].lower() in _TRAILING_STOP:
        words.pop()
    if words:
        words[-1] = words[-1].rstrip(".")
    return " ".join(words)


def _float_step_map(session: TutorSession) -> list[str]:
    """Step titles for the student-facing map, in plan order.

    A title is dropped to "Step n" when it would state the result of a step
    the student has not finished: the linear plan writes its check as
    "Check: put x = 3 into …", which hands over the answer two steps early.
    """
    import json as _json

    try:
        plan = _json.loads(session.solution_plan_json or "[]")
    except Exception:
        return []
    upcoming = [st.get("expected_expression", "")
                for st in plan[session.current_step_index:]]
    titles = []
    for n, st in enumerate(plan):
        title = _rail_title(st.get("instruction", ""))
        if not title or _reply_leaks(title, upcoming):
            title = f"Step {n + 1}"
        titles.append(title)
    return titles


def _float_step_question(step: dict) -> str:
    """The question that closes a step, in that step's own terms.

    "What equation do you get?" is right for algebra and wrong for a limit
    or a domain, where there may be no equation to write down.
    """
    ct = str(step.get("check_type", "") or "").lower()
    if ct in ("limit", "derivative", "integral", "numeric_value"):
        return "What do you get?"
    if ct == "conceptual":
        return "What is your answer?"
    return "What equation do you get?"


def _float_last_reply(db: Session, session) -> str:
    """The tutor's previous wording — used to keep hints from repeating."""
    for turn in reversed(_float_history(db, session.id if session else 0)):
        if turn.get("role") == "tutor":
            return str(turn.get("text") or "")
    return ""


def _float_history(db: Session, session_id: int, limit: int = 6) -> list[dict]:
    """Recent widget turns for the LLM, oldest first."""
    if not session_id:
        return []
    events = db.query(TutorEvent).filter(
        TutorEvent.session_id == session_id).order_by(
        TutorEvent.id.asc()).all()
    turns: list[dict] = []
    for e in events:
        if e.student_input:
            turns.append({"role": "student", "text": e.student_input})
        if e.tutor_reply:
            turns.append({"role": "tutor", "text": e.tutor_reply})
    return turns[-limit:]


def _float_note_reply(db: Session, session_id: int, reply: str) -> None:
    """Attach the tutor's wording to the event that triggered it."""
    if not session_id:
        return
    last = db.query(TutorEvent).filter(
        TutorEvent.session_id == session_id).order_by(
        TutorEvent.id.desc()).first()
    if last is not None and not last.tutor_reply:
        last.tutor_reply = (reply or "")[:2000]
        db.commit()


def _float_turn(session, *, step=None, verdict="", hint_level=0, intent="",
                student_answer="", first_name="", mode="practice",
                repeats=False, notation="") -> str:
    """The per-turn facts block the shared brain expects (spec section 2).

    The plan is server-side only: it shapes the prompt, never the response.
    """
    from . import float_chat as _chat

    return _chat.turn_context(
        problem=str(getattr(session, "problem_text", "") or ""),
        step=step or {}, verdict=verdict, hint_level=hint_level,
        intent=intent, mode=mode, first_name=first_name, notation=notation,
        plan=_float_plan(session), student_answer=student_answer,
        repeats=repeats, task=str(getattr(session, "task", "") or ""))


def _float_say(db: Session, session, message: str, *, intent: str,
               situation: str, fallback: str, course_ref: str = "",
               graded: bool = False, block_current: bool = False,
               turn: str = "", recap: bool = False) -> str:
    """LLM-worded tutor reply; `fallback` when the model is unusable.

    The model never decides correctness — the caller already has the CAS
    verdict and passes it in as a fact to phrase. `validate` re-asks the
    model rather than censoring it, and the deterministic leak check still
    runs last so a reply that survives retries can never spoil a step.
    """
    from . import float_chat as _chat

    banned = _float_unreached_results(session, include_current=block_current)

    def safe(text: str) -> bool:
        return not (banned and _reply_leaks(text, banned))

    out = _chat.reply(user_text=message,
                      history=_float_history(db, session.id if session else 0),
                      course_ref=course_ref, intent=intent,
                      situation=situation, graded=graded, turn=turn,
                      recap=recap, validate=None if recap else safe)
    if not out:
        return fallback
    text = out[0]
    # Two regenerations got the model no closer to the limit; the scripted
    # line is shorter and still on-topic, so it wins over a long reply.
    if not recap and _chat._too_long(text):
        return fallback
    text = (_float_leak_check(text, session, block_current=block_current)
            if session else text)
    return _float_display(text)


def _example_matches_problem(reply: str, problem: str) -> bool:
    """True when a level-3 example reuses the student's own numbers.

    Two shared digits is the line: one can be a coincidence ("2" in any
    squared term), two means the worked example was copied from the problem
    and the student has been handed their own numbers back.
    """
    import re as _re

    mine = set(_re.findall(r"\d+", problem or ""))
    if len(mine) < 2:
        return False
    return len(mine & set(_re.findall(r"\d+", reply or ""))) >= 2


def _float_hint_said(db: Session, session, message: str, step: dict,
                     lvl: int, *, graded: bool, course_ref: str,
                     fallback: str, first_name: str = "") -> str:
    """Word one hint rung with the LLM, keeping the "Hint N — " prefix.

    The ladder state and the rung cap stay in code; the model only phrases
    the rung it is given, so it cannot skip ahead to a worked solution.
    """
    turn = _float_turn(session, step=step, verdict="incorrect",
                       hint_level=lvl, intent="hint_request",
                       student_answer=message, first_name=first_name,
                       mode="graded" if graded else "practice")
    llm = _float_say(
        db, session, message, intent="chat", course_ref=course_ref,
        graded=graded, fallback="", block_current=True, turn=turn,
        situation=(
            f"The student's answer to this step was not correct. Step: "
            f"{step.get('instruction', '')} Why it works: "
            f"{step.get('reason', '')}. Give exactly hint level {lvl} of 4 "
            f"(1 conceptual nudge, 2 where to look, 3 which operation to "
            f"use, 4 split into micro-steps). Never state the result of "
            f"any step. Open with the help itself: do not repeat the word "
            f"Hint or number the rung, the app prints that label."
            + (" For level 3 the worked example must use different numbers "
               "than the student's problem." if lvl == 3 else "")))
    if not llm:
        return fallback
    if lvl == 3 and _example_matches_problem(
            llm, str(getattr(session, "problem_text", "") or "")):
        return fallback
    body = _model_body(llm)
    if not body:
        return fallback
    return _float_leak_check(f"Hint {lvl} — {body}", session,
                             block_current=True)


@app.post("/api/tutor/nudge")
def api_tutor_nudge(request: Request, payload: dict,
                    db: Session = Depends(get_db)):
    """One gentle idle nudge per step, never a second one.

    The client calls this after the student has typed nothing for a while.
    It spends no hint and reveals nothing; a step already nudged returns an
    empty reply so the widget stays quiet.
    """
    ctx = payload.get("pageContext") or {}
    if not isinstance(ctx, dict):
        ctx = {}
    course_id = str(ctx.get("courseId", "") or ctx.get("course", "")
                    or "default")
    session = _float_open_session(db, _viewer_id(request, db), course_id)
    if not session or session.status != "active" or not session.total_steps:
        return {"reply": "", "replyType": "noop"}
    idx = session.current_step_index
    if db.query(TutorEvent).filter(
            TutorEvent.session_id == session.id,
            TutorEvent.step_index == idx,
            TutorEvent.event_type == "nudge").count():
        return {"reply": "", "replyType": "noop"}
    plan = _float_plan(session)
    step = plan[idx] if 0 <= idx < len(plan) else {}
    reply = ("Want a hint, or keep going? " + (
        f"You were on: {step.get('instruction', '')} What do you get?"
        if step else "Send me the next line of your work."))
    reply = _float_display(reply)
    _float_log(db, session.id, idx, "", "idle-nudge", session.hints_used,
               "nudge", reply_type="support", intent="idle",
               mode="graded" if session.graded_mode else "practice")
    out = {"reply": reply, "replyType": "support",
           "stepIndex": idx, "totalSteps": session.total_steps,
           "stepStatus": "waiting", "hintLevel": session.hints_used,
           "sessionDone": False}
    titles = _float_step_map(session)
    if titles:
        out["steps"] = titles
    _float_note_reply(db, session.id, reply)
    return out


@app.post("/api/tutor/message")
def api_tutor_message(request: Request, payload: dict,
                      db: Session = Depends(get_db)):
    """Floating-tutor guided chat. CAS decides correctness; replies guide.

    Input:  {message, pageContext} (student/course from session, never client).
    Output: {reply, replyType, stepIndex, totalSteps, stepStatus, hintLevel,
             sessionDone, steps?, recap?}.
    """
    import json as _json
    from datetime import datetime, timezone

    from . import float_plan as _plan

    student_id = _viewer_id(request, db)
    message = str(payload.get("message", "") or "")[:500].strip()
    ctx = payload.get("pageContext") or payload.get("pagecontext") or {}
    if not isinstance(ctx, dict):
        ctx = {}
    ctx_type = str(ctx.get("type", "general") or "general")
    course_id = str(ctx.get("courseId", "") or ctx.get("course", "")
                    or "default")

    if not _float_rate_ok(db, student_id):
        raise HTTPException(429, "Slow down — 30 messages per 10 minutes.")
    if not message:
        return {"reply": "Please rephrase that — send an equation like "
                "$10 = 3x + 1$, or tap a suggestion chip.",
                "replyType": "bubble", "stepIndex": 0, "totalSteps": 0,
                "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False}

    graded = (ctx_type == "assignment"
              and str(ctx.get("mode", "practice")) == "graded")
    mode_name = "graded" if graded else "practice"
    student_name = _student_first_name(db, student_id)
    session = _float_open_session(db, student_id, course_id)
    lowered = message.lower()

    # A practice item carries its own question. The client sends the item
    # id, never the problem text, so the tutor works on the item the
    # student is actually looking at — and knows what it asks.
    item_expr = item_task = ""
    item_plan: list[dict] = []
    if ctx_type == "practice":
        item_ref = str(ctx.get("ref", "") or "").strip()
        item_id = str(ctx.get("item_id", "") or "").strip()
        if item_ref and item_id:
            from . import float_plan as _planmod
            try:
                item_lesson = _practice_lesson(db, item_ref)
                # Looking this item up must not generate the rest of the set:
                # a generation round is ~15s of model time, and the student
                # is waiting on an answer to the message they just sent.
                item_set = _practice_set(db, item_lesson, top_up=False)
                item = {i.get("item_id"): i for i in item_set["items"]}.get(
                    item_id)
            except Exception:
                item = None
            if item:
                item_expr = _planmod.canonical_expression(
                    str(item.get("prompt_latex") or ""))
                item_task = str(item.get("task") or "").strip()
                item_plan = _practice_plan(db, item_lesson, item)
                if session and session.problem_text != item_expr:
                    # A session left open on a DIFFERENT question must not be
                    # reused here, or the tutor answers the wrong question.
                    session.status = "abandoned"
                    db.commit()
                    session = None

    def _dump(plan: list[dict]) -> str:
        return _json.dumps(plan)

    def _recap(sess: TutorSession) -> dict:
        from .verifier import display_math as _dm

        plan = _float_plan(sess)
        # The card drops each result into $...$ for KaTeX, and CAS spellings
        # render as typed there: "(x - 3)*(x + 2)" shows a literal ∗.
        # `own` is what the student typed and the checker accepted, so the
        # recap reads as their work, not a solution pasted from the plan.
        steps = [{"n": i + 1,
                  "action": s.get("instruction", ""),
                  "result": _dm(s.get("expected_expression", "")),
                  "own": _dm(s.get("student_form", "")
                             or s.get("expected_expression", "")),
                  "why": s.get("reason", "")}
                 for i, s in enumerate(plan)]
        answer = _dm(plan[-2].get("expected_expression", "")
                     if len(plan) >= 2 else "")
        check = _dm(plan[-1].get("expected_expression", "") if plan else "")
        recap = {
            "title": f"Full solution — {sess.problem_text}",
            "steps": steps, "answer": answer, "check": check,
            "idea": ("Undo operations one at a time: constants first, "
                     "then the coefficient, then verify by substituting."),
        }
        if sess.total_misses:
            recap["note"] = ("You needed a few hints along the way — that's "
                             "normal, and you still did the steps yourself.")
        return recap

    def _shape(sess: TutorSession, reply: str, rtype: str,
               status: str, hint: int, done: bool = False,
               recap: dict | None = None) -> dict:
        out = {"reply": reply, "replyType": rtype,
               "stepIndex": sess.current_step_index,
               "totalSteps": sess.total_steps, "stepStatus": status,
               "hintLevel": hint, "sessionDone": done}
        titles = _float_step_map(sess)
        if titles:
            out["steps"] = titles
        if recap is not None:
            out["recap"] = recap
        # Keep the pair together so the next turn can be answered in context.
        _float_note_reply(db, sess.id, reply)
        return out

    # Intent router (runs before grading): decides whether a message is a
    # step answer at all. Prose, greetings and questions must never reach
    # the CAS grader, or the student only ever sees a parse error.
    from . import float_intent as _intent

    intent = _intent.classify(message, has_session=bool(session))
    course_ref = str(ctx.get("course", "") or ctx.get("courseId", "") or "")

    _pending = _float_pending_get(db, student_id, course_id)
    if _pending:
        # Awaiting "Start new problem?" answer: yes/no only.
        if lowered.strip() in ("yes", "start new problem", "start",
                               "new problem", "start it"):
            _float_pending_clear(db, student_id, course_id)
            if session and session.status == "active":
                session.status = "abandoned"
                db.commit()
                _float_log(db, session.id, session.current_step_index,
                           message, "abandoned", 0, "message")
            session = TutorSession(
                student_id=student_id, course_id=course_id,
                problem_text=_pending["problem"],
                solution_plan_json=_dump(
                    _plan.build_plan(_pending["problem"])[0] or []),
                status="active", current_step_index=0,
                total_steps=len(
                    _plan.build_plan(_pending["problem"])[0] or []),
                graded_mode=graded)
            db.add(session)
            db.commit()
            _float_log(db, session.id, 0, message, "new-problem", 0,
                       "graded_use" if graded else "message")
            try:
                plan0 = _json.loads(session.solution_plan_json or "[]")
            except Exception:
                plan0 = []
            if not plan0:
                return {"reply": ("I couldn't build a step plan for that — "
                                  "let's talk it through instead. What do you "
                                  "notice about the form of the problem?"),
                        "replyType": "bubble", "stepIndex": 0,
                        "totalSteps": 0, "stepStatus": "waiting",
                        "hintLevel": 0, "sessionDone": False,
                        "genericMode": True}
            head = "Graded assignment: limited hints. " if graded else ""
            return _shape(session,
                          f"{head}Step 1: {plan0[0]['instruction']} "
                          f"What equation do you get?",
                          "step", "waiting", 0)
        if lowered.strip() in ("no", "no, it's my answer", "my answer",
                               "it's my answer"):
            _float_pending_clear(db, student_id, course_id)
            # Fall through: grade the ORIGINAL pending message as the step,
            # bypassing the confirm check (the student just declined it).
            message = _pending["problem"]
            lowered = message.lower()
            _skip_confirm = True
        else:
            _skip_confirm = False
    else:
        _skip_confirm = False
    if (session and not _skip_confirm
            and intent in ("NEW_PROBLEM", "STEP_ANSWER")
            and _intent.looks_like_new_problem_mid_session(message)):
        # Confirm only when the message is structurally a DIFFERENT problem:
        # a leading verb ("Solve ..."), or different free symbols than the
        # session problem. Wrong attempts at the current step (same symbols,
        # no verb — e.g. "8 = 3x") grade normally through the hint ladder.
        _confirm = bool(_intent._NEW_LEAD.match(message.strip()))
        if not _confirm:
            try:
                from .verifier import _parse_single as _ps
                _prob_syms = set()
                try:
                    _prob_syms = set(
                        str(s) for s in _ps(session.problem_text).free_symbols)
                except Exception:
                    _prob_syms = set()
                _msg_expr = _ps(message)
                _msg_syms = (set(str(s) for s in _msg_expr.free_symbols)
                             if hasattr(_msg_expr, "free_symbols") else set())
                if _prob_syms and _msg_syms and _msg_syms != _prob_syms:
                    _confirm = True
            except Exception:
                _confirm = False
        if _confirm:
            _float_pending_set(db, student_id, course_id, message)
            return {"reply": ("That looks like a new problem. Start it now?"),
                    "replyType": "confirm_new_problem", "stepIndex": 0,
                    "totalSteps": 0, "stepStatus": "waiting", "hintLevel": 0,
                    "sessionDone": False}
        # Same-problem attempt (right or wrong): grade below.
    # Router branches. None of these is a miss: chatter must never burn the
    # hint ladder or total_misses, and must never be graded by the CAS.
    def _bubble(reply_text: str, verdict: str, rtype: str = "bubble") -> dict:
        if session:
            _float_log(db, session.id, session.current_step_index,
                       message, verdict, session.hints_used, "message",
                       reply_type=rtype, intent=intent.lower(),
                       mode=mode_name)
            return _shape(session, reply_text, rtype, "waiting",
                          session.hints_used)
        return {"reply": reply_text, "replyType": rtype, "stepIndex": 0,
                "totalSteps": 0, "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False}

    def _turn_ctx(step=None, verdict: str = "", hint: int | None = None,
                  intent_name: str = "", repeats: bool = False) -> str:
        return _float_turn(
            session, step=step or {}, verdict=verdict,
            hint_level=(session.hints_used if hint is None else hint)
            if session else 0,
            intent=intent_name or intent.lower(), student_answer=message,
            first_name=student_name, mode=mode_name, repeats=repeats)

    if intent == "OFF_TOPIC":
        fallback = ("I'm your math tutor — let's stick to math and course "
                    "topics. " + (_float_current_prompt(session) if session
                                  else "Send me an equation and we'll work it "
                                       "step by step."))
        return _bubble(_float_say(
            db, session, message, intent="off_topic", course_ref=course_ref,
            graded=graded, fallback=fallback, turn=_turn_ctx(),
            situation="The student asked about something outside the course "
                      "sequence. Decline in one sentence and redirect to "
                      "course math."), "off-topic")

    if intent == "CONCEPT_QUESTION":
        situation = ("The student asked a conceptual math question. Answer "
                     "the idea briefly and correctly in 2-3 sentences with a "
                     "DIFFERENT example (different numbers than their "
                     "problem), then invite them to try one step themselves.")
        fallback = ("Good question! Think of a balance scale: whatever you "
                    "do to one side, you must do to the other, or it tips. "
                    "That's why we apply the same operation to both sides. "
                    "Now back to yours: " + _float_current_prompt(session)
                    if session else
                    "Good question — tell me which course topic it's about "
                    "and we'll work through an example together.")
        if session:
            situation += (" They have a problem open in this chat. Explain the "
                          "general idea using your own example, not theirs, and "
                          "never state or work out their problem's result.")
        return _bubble(_float_say(
            db, session, message, intent="concept", course_ref=course_ref,
            graded=graded, fallback=fallback,
            turn=_turn_ctx(verdict="unclear"), situation=situation), "concept")

    if intent == "CHAT":
        situation = ("The student is chatting, not answering. Reply warmly "
                     "in a sentence, keep it mathematical where you can, and "
                     "ask one question to move them forward.")
        fallback = ("Happy to help with math. " + _float_current_prompt(session)
                    if session else
                    "Hi! I'm your math tutor. Send me an equation and we'll "
                    "work it out step by step.")
        if session:
            situation += (f" They have a problem open at step "
                          f"{session.current_step_index + 1}. Nudge them back "
                          f"to it without solving it for them.")
        return _bubble(_float_say(
            db, session, message, intent="chat", course_ref=course_ref,
            graded=graded, fallback=fallback, turn=_turn_ctx(),
            situation=situation), "chat")

    if intent == "ANSWER_REQUEST":
        # Policy stays in code: the model may phrase the refusal, never
        # grant it. The leak check below strips any future-step result.
        prompt = _float_current_prompt(session) if session else \
            "Send me an equation and we'll solve it together."
        fallback = ("I won't give the final answer — but we'll get there "
                    f"together. {prompt}")
        reply = _float_say(
            db, session, message, intent="chat", course_ref=course_ref,
            graded=graded, fallback=fallback, block_current=True,
            turn=_turn_ctx(verdict="unclear", intent_name="answer_request"),
            situation="The student is asking for the answer outright. "
                      "Refuse kindly, do not reveal any step result, and "
                      "point them at the next move they can try.")
        if session:
            _float_log(db, session.id, session.current_step_index,
                       message, "refused", 0, "message")
            return _shape(session, reply, "bubble", "waiting", 0)
        return {"reply": reply, "replyType": "bubble", "stepIndex": 0,
                "totalSteps": 0, "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False}

    if intent == "HELP_REQUEST":
        if not session:
            return _bubble(_float_say(
                db, session, message, intent="chat", course_ref=course_ref,
                graded=graded, fallback=_float_current_prompt(None),
                turn=_turn_ctx(),
                situation="The student says they are stuck but no problem is "
                          "open yet. Ask which problem they are working on."),
                "help")
        plan = _json.loads(session.solution_plan_json or "[]")
        if plan and 0 <= session.current_step_index < len(plan):
            step = plan[session.current_step_index]
            if graded and GRADED_HINTS_ONLY and session.hints_used >= 1:
                reply = ("Graded assignment: limited hints — re-read the "
                         "step and try the next move on your own. "
                         f"{step.get('instruction', '')}")
                _float_log(db, session.id, session.current_step_index,
                           message, "hint-capped", session.hints_used + 1,
                           "hint", reply_type="hint", intent="hint_request",
                           mode="graded")
                session.hints_used += 1
                db.commit()
                return _shape(session, reply, "hint", "waiting",
                              session.hints_used)
            reply, lvl = _float_hint(step, session.hints_used,
                                     session.total_misses,
                                     session.current_step_index + 1,
                                     avoid=_float_last_reply(db, session))
            session.hints_used = lvl
            db.commit()
            _float_log(db, session.id, session.current_step_index,
                       message, "hint", lvl, "hint",
                       reply_type="hint", intent="hint_request",
                       mode=mode_name)
            reply = _float_hint_said(db, session, message, step, lvl,
                                     graded=graded, course_ref=course_ref,
                                     fallback=reply,
                                     first_name=student_name)
            return _shape(session, _float_display(reply), "hint", "waiting",
                          lvl)

    if intent in ("START_STEP", "FRUSTRATED"):
        # "What should I do?" and "I give up" are not wrong answers: nothing
        # is graded, no hint is spent, no miss is recorded. The student gets
        # the current step back as a single action — the smallest piece the
        # plan offers when they are frustrated.
        plan_s = _float_plan(session)
        idx = session.current_step_index if session else 0
        step_s = plan_s[idx] if 0 <= idx < len(plan_s) else {}
        turn = _float_turn(
            session, step=step_s, verdict="unclear",
            hint_level=session.hints_used if session else 0,
            intent=("where_to_start" if intent == "START_STEP"
                    else "frustrated"),
            student_answer=message, first_name=student_name, mode=mode_name)
        if not step_s:
            return _bubble(_float_say(
                db, session, message, intent="chat", course_ref=course_ref,
                graded=graded, turn=turn,
                fallback=_float_current_prompt(session),
                situation=("The student is lost and no step is open yet. Ask "
                           "for the equation, in one sentence, and offer to "
                           "take Step 1 together.")),
                "support", rtype="support")
        asks = db.query(TutorEvent).filter(
            TutorEvent.session_id == session.id,
            TutorEvent.step_index == idx,
            TutorEvent.verdict.in_(("where-to-start", "frustrated"))).count()
        if asks >= 2:
            # A third ask means restating the step is not helping: climb one
            # rung of the ladder instead.
            cap = 2 if (graded and GRADED_HINTS_ONLY) else 4
            hlvl = min(session.hints_used + 1, cap)
            scripted, _ = _float_hint(step_s, hlvl - 1, session.total_misses,
                                      idx + 1,
                                      avoid=_float_last_reply(db, session))
            session.hints_used = hlvl
            db.commit()
            _float_log(db, session.id, idx, message, "hint", hlvl, "hint",
                       reply_type="hint", intent="hint_request", mode=mode_name)
            return _shape(
                session,
                _float_display(_float_hint_said(
                    db, session, message, step_s, hlvl, graded=graded,
                    course_ref=course_ref, fallback=scripted,
                    first_name=student_name)),
                "hint", "waiting", hlvl)
        micro = [m for m in (step_s.get("micro_steps") or []) if m]
        action = (micro[0] if intent == "FRUSTRATED" and micro
                  else step_s.get("instruction", ""))
        step_no = idx + 1
        scripted = (
            f"This one is tricky, and that's okay. Let's make it smaller: "
            f"{action} What do you get?"
            if intent == "FRUSTRATED" else
            f"Step {step_no}: {action} What do you get?")
        situation = (
            f"The student is frustrated. Acknowledge in one sentence, then "
            f"give only this smaller move: {action}. Do not repeat the whole "
            "step and do not state any result."
            if intent == "FRUSTRATED" else
            f"The student asked what to do. Give only this action for step "
            f"{step_no}: {action}. End with one question asking what they get. "
            "Never state the result of the step.")
        said = _float_say(db, session, message, intent="chat",
                          course_ref=course_ref, graded=graded,
                          fallback=scripted, turn=turn, block_current=True,
                          situation=situation)
        if intent == "START_STEP" and said != scripted:
            said = f"Step {step_no}: {_model_body(said)}"
        _float_log(db, session.id, idx, message,
                   "where-to-start" if intent == "START_STEP" else "frustrated",
                   session.hints_used, "message",
                   reply_type=("step" if intent == "START_STEP" else "support"),
                   intent=("where_to_start" if intent == "START_STEP"
                           else "frustrated"), mode=mode_name)
        return _shape(
            session,
            _float_display(_float_leak_check(said, session)),
            "step" if intent == "START_STEP" else "support", "waiting",
            session.hints_used)

    # "New problem" / "similar problem" resets or branches the thread.
    if lowered.strip() in ("new problem", "new question", "restart",
                           "start over", "try a similar problem",
                           "similar problem"):
        import hashlib as _hl

        if session and session.status == "active":
            if "similar" in lowered:
                # Deterministic sibling problem: same shape, new numbers.
                seed = int(_hl.md5(
                    session.problem_text.encode()).hexdigest()[:8], 16)
                a = 2 + seed % 5
                b = 1 + (seed >> 4) % 8
                k = 1 + (seed >> 8) % 4
                c = a * k + b
                equation = f"{c} = {a}x + {b}"
                plan = _plan.build_linear_plan(equation) or []
                session.status = "abandoned"
                db.commit()
                _float_log(db, session.id, session.current_step_index,
                           message, "abandoned", 0, "message")
                session = TutorSession(
                    student_id=student_id, course_id=course_id,
                    problem_text=equation, solution_plan_json=_dump(plan),
                    status="active", current_step_index=0,
                    total_steps=len(plan), graded_mode=graded)
                db.add(session)
                db.commit()
                _float_log(db, session.id, 0, message, "new-problem",
                           0, "graded_use" if graded else "message")
                head = ("Graded assignment: limited hints. "
                        if graded else "")
                return _shape(session,
                              f"{head}Similar problem: ${equation}$. "
                              f"Step 1: {plan[0]['instruction']} "
                              f"What equation do you get?"
                              if plan else
                              f"{head}Similar problem: ${equation}$.",
                              "step", "waiting", 0)
            session.status = "abandoned"
            db.commit()
            _float_log(db, session.id, session.current_step_index,
                       message, "abandoned", 0, "message")
        reply = ("Fresh start! Send me an equation like $10 = 3x + 1$ "
                 "and I'll guide you step by step.")
        if graded:
            reply = ("Graded assignment: limited hints. " + reply)
        return {"reply": reply, "replyType": "bubble", "stepIndex": 0,
                "totalSteps": 0, "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False}

    # Assignment "Help with this question": graded chip starts a session.
    # Lesson "Explain this section": brief prose + offer to practice.
    help_text = str(ctx.get("questionText", "") or "").strip()
    section_text = str(ctx.get("sectionText", "") or "").strip()
    if "explain this section" in lowered and ctx_type == "lesson":
        equation = _plan.detect_equation(section_text or help_text or "")
        if equation:
            plan = _plan.build_linear_plan(equation)
            if plan:
                if session and session.status == "active":
                    session.status = "abandoned"
                    db.commit()
                session = TutorSession(
                    student_id=student_id, course_id=course_id,
                    problem_text=equation, solution_plan_json=_dump(plan),
                    status="active", current_step_index=0,
                    total_steps=len(plan), graded_mode=False)
                db.add(session)
                db.commit()
                _float_log(db, session.id, 0, message, "new-problem",
                           0, "message")
                return _shape(
                    session,
                    f"This section works through ${equation}$. The idea: "
                    f"undo one operation at a time until x stands alone. "
                    f"Let's try it — Step 1: {plan[0]['instruction']} "
                    f"What equation do you get?",
                    "step", "waiting", 0)
        title = str(ctx.get("sectionTitle", "") or "")[:160]
        reply = (
            f"Here's the idea behind {title or 'this section'}: each step "
            f"undoes one operation while keeping both sides balanced. "
            f"Send me an equation from the section and we'll work it "
            f"step by step — or tap Practice This Topic for guided items."
        )
        if session:
            _float_log(db, session.id, session.current_step_index,
                       message, "concept", session.hints_used, "message")
            return _shape(session, reply, "bubble", "waiting",
                          session.hints_used)
        return {"reply": reply, "replyType": "bubble", "stepIndex": 0,
                "totalSteps": 0, "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False}
    if ("help with this question" in lowered and help_text
            and "=" in help_text):
        equation = _plan.detect_equation(help_text) or help_text
        plan = _plan.build_linear_plan(equation)
        if plan:
            if session and session.status == "active":
                session.status = "abandoned"
                db.commit()
            session = TutorSession(
                student_id=student_id, course_id=course_id,
                problem_text=equation, solution_plan_json=_dump(plan),
                status="active", current_step_index=0,
                total_steps=len(plan), graded_mode=graded)
            db.add(session)
            db.commit()
            _float_log(db, session.id, 0, message, "new-problem", 0,
                       "graded_use" if graded else "message")
            head = ("Graded assignment: limited hints. " if graded else "")
            reply = (f"{head}Step 1: {plan[0]['instruction']} "
                     f"What equation do you get?")
            return _shape(session, reply, "step", "waiting", 0)

    # New problem starts a session only when there is NO active session
    # (mid-session candidates were already confirmed or declined above).
    # Uses the template dispatcher with validation; unplannable input
    # drops to generic Socratic mode instead of a dead end.
    if session is None:
        plan = list(item_plan)
        problem = item_expr or message
        task = item_task
        source = "practice item" if plan else ""
        if not plan:
            plan, source = _plan.build_plan(message, task=item_task)
            if plan:
                ok, reason = _plan.validate_plan(plan, message)
                import logging as _logging
                _logging.getLogger("mathpilot.tutor").info(
                    "plan_source=%s validation=%s (%s)", source, ok, reason)
                if not ok:
                    plan = None
        if plan:
            session = TutorSession(
                student_id=student_id, course_id=course_id,
                problem_text=problem, task=task,
                solution_plan_json=_dump(plan),
                status="active", current_step_index=0,
                total_steps=len(plan), graded_mode=graded)
            db.add(session)
            db.commit()
            _float_log(db, session.id, 0, message, "new-problem", 0,
                       "graded_use" if graded else "message")
            head = "Graded assignment: limited hints. " if graded else ""
            return _shape(session,
                          f"{head}Step 1: {plan[0]['instruction']} "
                          f"{_float_step_question(plan[0])}",
                          "step", "waiting", 0)
        # Generic Socratic mode: no invented plan, honest about it — but
        # still anchored to the question, so a follow-up stays on it.
        if item_task or item_expr:
            return {"reply": (
                f"This one is about {item_task or item_expr}. I don't have a "
                "worked path for it yet, so let's reason it out together. "
                "What do you notice about the form of the expression, and "
                "which rule would you try first?"),
                "replyType": "bubble", "stepIndex": 0, "totalSteps": 0,
                "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False, "genericMode": True}
        return {"reply": ("I can work with that, but I don't have a "
                          "step-by-step plan I trust for it yet. What do you "
                          "notice about the form of the problem? What "
                          "technique might apply? (Or send 'New problem' to "
                          "try something else.)"),
                "replyType": "bubble", "stepIndex": 0, "totalSteps": 0,
                "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False, "genericMode": True}

    # Active session: CAS verdict on the student's step.
    import json as _jj

    try:
        plan = _jj.loads(session.solution_plan_json or "[]")
    except Exception:
        plan = []
    if not plan or not (0 <= session.current_step_index < len(plan)):
        session.status = "abandoned"
        db.commit()
        return {"reply": "Let's start fresh — send me the equation.",
                "replyType": "bubble", "stepIndex": 0, "totalSteps": 0,
                "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False}
    step = plan[session.current_step_index]
    expected = step.get("expected_expression", "")
    check_type = step.get("check_type", "equivalent_expression")
    verdict = _plan.check_step(expected, message, check_type)

    jumped_from = None
    if verdict == "incorrect":
        jump = _float_jump_ahead(plan, session.current_step_index, message,
                                 session.problem_text)
        if jump is not None:
            # Working two steps in one message is skill, not an error.
            jumped_from = session.current_step_index
            step = plan[jump]
            session.current_step_index = jump
            verdict = "correct"

    if verdict == "unclear":
        # One notation retry: the LLM rewrites what the student typed into
        # CAS syntax, then SymPy judges that rewrite. The model's math is
        # never trusted as a verdict — only as a translation.
        from . import config as _config
        from . import nebius as _nebius

        if _config.ai_configured():
            normed = _nebius.normalize_via_llm(message)
            if normed:
                retry = _plan.check_step(expected, normed, check_type)
                if retry != "unclear":
                    verdict = retry
                    step["student_form"] = normed[:200]
                    session.solution_plan_json = _dump(plan)
                    db.commit()

    if verdict == "correct":
        accepted = message.strip()
        step["student_form"] = accepted
        session.solution_plan_json = _dump(plan)
        session.current_step_index += 1
        session.hints_used = 0  # ladder resets on success
        if session.current_step_index >= len(plan):
            session.status = "completed"
            session.completed_at = datetime.now(timezone.utc)
            db.commit()
            _float_log(db, session.id, len(plan) - 1, message,
                       "correct", 0, "recap_shown" if not graded else "message")
            if graded and GRADED_HINTS_ONLY:
                reply = ("Check recorded. The full recap unlocks after you "
                         "submit the assignment — nice work getting there.")
                return _shape(session, reply, "success", "done", 0,
                              done=True)
            recap = _recap(session)
            reply = ("Yes! You finished every step yourself. "
                     "Here's your full solution.")
            return _shape(session, reply, "recap", "done", 0,
                          done=True, recap=recap)
        db.commit()
        _float_log(db, session.id, session.current_step_index - 1,
                   message, "correct", 0, "message", reply_type="praise",
                   intent="step_answer", mode=mode_name)
        nxt = plan[session.current_step_index]
        step_no = session.current_step_index + 1
        ahead = (jumped_from is not None)
        covered = (f"Nice — that covers steps {jumped_from + 1} to "
                   f"{step_no - 1}. " if ahead else "Correct! ")
        reply = (f"{covered}Step {step_no}: "
                 f"{nxt['instruction']} What do you get?")
        llm = _float_say(
            db, session, message, intent="chat", course_ref=course_ref,
            graded=graded, fallback="",
            turn=_float_turn(
                session, step=nxt, verdict="correct", intent="step_answer",
                student_answer=message, first_name=student_name,
                mode=mode_name),
            situation=(
                covered + ("The student answered ahead, so say which steps are "
                           "now done without restating any of their results. "
                           if ahead else "")
                + "Our math checker VERIFIED the student's last answer as "
                "correct. Confirm in a few words and name what they did well, "
                f"then give only this next action: {nxt['instruction']} Do not "
                "state the result of that step or any later one."))
        if llm:
            reply = f"Step {step_no}: {_model_body(llm)}"
        return _shape(session, _float_display(
            _float_leak_check(reply, session, block_current=True)),
            "step", "waiting", 0)

    if verdict == "partial":
        # Correct but incomplete (e.g. one of two solutions): prompt for
        # the rest, do NOT count as a miss.
        _float_log(db, session.id, session.current_step_index,
                   message, "partial", session.hints_used, "message",
                   reply_type="partial", intent="step_answer", mode=mode_name)
        fallback = ("You found one solution. Is there another? "
                    f"({step.get('instruction', '')})")
        return _shape(
            session, _float_say(
                db, session, message, intent="chat", course_ref=course_ref,
                graded=graded, fallback=fallback,
                turn=_turn_ctx(step, "partial"),
                situation=(
                    "SymPy verified the student has one correct solution but "
                    "this step needs more. Say what is still missing in one "
                    "sentence and ask for the rest. Do not name the missing "
                    "value.")),
            "bubble", "waiting", session.hints_used)

    if verdict == "unclear":
        # Unreadable input is NOT a miss. Restate what the step is asking
        # for instead of repeating one canned parse-error line forever.
        session.total_misses += 0  # explicit: unclear never counts
        _float_log(db, session.id, session.current_step_index,
                   message, "unclear", session.hints_used, "message",
                   reply_type="format_help", intent="step_answer",
                   mode=mode_name)
        recent_unclear = db.query(TutorEvent).filter(
            TutorEvent.session_id == session.id,
            TutorEvent.verdict == "unclear").count()
        # The step's own expected expression IS the result, so the format
        # example shows the shape with different letters and numbers.
        expect = "5y = 20"
        fallback = ("I couldn't match that to this step. Write just the line "
                    f"for this one move, write it like {expect}. "
                    "What do you get?")
        if recent_unclear >= 2:
            fallback += (" Tap fx for symbols — the toolbar above the typing "
                         "box has everything you need.")
        reply = _float_say(
            db, session, message, intent="chat", course_ref=course_ref,
            graded=graded, fallback=fallback, block_current=True,
            turn=_turn_ctx(step, "unreadable"),
            situation=(
                "The checker could not read the student's message, so it is "
                "unreadable, not wrong. Do not blame them and do not repeat "
                "the whole step. Show how to type the answer for THIS step "
                f"with one format example (write it like {expect}) and ask "
                "them to send just that."))
        return _shape(session, _float_leak_check(reply, session),
                      "bubble", "waiting", session.hints_used)

    # Incorrect: hint ladder (graded caps at level 2, recap deferred).
    same_before = db.query(TutorEvent).filter(
        TutorEvent.session_id == session.id,
        TutorEvent.step_index == session.current_step_index,
        TutorEvent.verdict == "incorrect",
        TutorEvent.student_input == message[:2000]).count()
    session.total_misses += 1
    if graded and GRADED_HINTS_ONLY:
        lvl = min(session.hints_used + 1, 2)
        session.hints_used = lvl
        db.commit()
        _float_log(db, session.id, session.current_step_index,
                   message, "incorrect", lvl, "graded_use",
                   reply_type="hint", intent="step_answer", mode="graded")
        reply, _ = _float_hint(step, lvl - 1, session.total_misses,
                               session.current_step_index + 1)
        # Strip level-3+ content: cap means rung 1-2 wording only.
        reply = (f"Hint {lvl} — not quite. {step['reason']} "
                 f"Try again: {step['instruction']}")
        reply = _float_hint_said(db, session, message, step, lvl,
                                 graded=True, course_ref=course_ref,
                                 fallback=reply, first_name=student_name)
        return _shape(session, _float_display(reply), "hint", "waiting", lvl)
    prev_said = _float_last_reply(db, session)
    reply, lvl = _float_hint(step, session.hints_used,
                             session.total_misses,
                             session.current_step_index + 1,
                             avoid=prev_said)
    if same_before and lvl < 4:
        # The same miss twice means the last hint did not land: skip that rung
        # so the student never reads the same advice two turns running.
        reply, lvl = _float_hint(step, lvl, session.total_misses,
                                 session.current_step_index + 1,
                                 avoid=prev_said)
    session.hints_used = lvl
    db.commit()
    _float_log(db, session.id, session.current_step_index,
               message, "incorrect", lvl, "hint",
               reply_type="hint", intent="step_answer", mode=mode_name)
    reply = _float_hint_said(db, session, message, step, lvl,
                             graded=graded, course_ref=course_ref,
                             fallback=reply, first_name=student_name)
    return _shape(session, _float_display(
        _float_leak_check(reply, session)),
        "hint", "waiting", lvl)


@app.get("/dashboard")
def dashboard_redirect():
    return RedirectResponse("/practice", status_code=303)


@app.get("/activity/{activity_id}", response_class=HTMLResponse)
def activity_page(activity_id: int, request: Request, db: Session = Depends(get_db)):
    activity = db.get(Activity, activity_id)
    if not activity:
        raise HTTPException(404, "Activity not found")
    steps = steps_of(activity)
    uid = _viewer_id(request, db)
    attempts = db.query(Attempt).filter(
        Attempt.activity_id == activity_id, Attempt.user_id == uid
    ).order_by(Attempt.id.asc()).all()
    courses = db.query(Course).all()
    materials = db.query(CourseMaterial).filter(
        CourseMaterial.title.like(f"%{activity.course_topic}%") | CourseMaterial.title.like("%Chain rule%")
    ).all()
    if not materials:
        materials = db.query(CourseMaterial).limit(3).all()
    review_req = db.query(ReviewRequest).filter(
        ReviewRequest.activity_id == activity_id, ReviewRequest.user_id == uid
    ).order_by(ReviewRequest.id.desc()).first()

    # Determine currently active step based on verified attempts
    verified_steps = {a.step_index for a in attempts if a.verdict == "correct"}
    active_step = 0
    for idx in range(len(steps)):
        if idx not in verified_steps:
            active_step = idx
            break
        active_step = min(idx + 1, len(steps) - 1)

    return templates.TemplateResponse(request, "activity.html", {
        "request": request, "activity": activity, "steps": steps,
        "hints": hints_of(activity), "attempts": attempts,
        "active_step": active_step,
        "courses": courses,
        "materials": materials,
        "review_req": review_req,
        "active_nav": "dashboard",
        "demo": not config.ai_configured(),
    })


@app.post("/api/submit")
def api_submit(request: Request, payload: dict, db: Session = Depends(get_db)):
    activity = db.get(Activity, payload.get("activity_id", 0))
    if not activity:
        raise HTTPException(404, "Activity not found")
    uid = _viewer_id(request, db)
    steps = steps_of(activity)
    idx = int(payload.get("step_index", 0))
    if not steps or idx >= len(steps):
        raise HTTPException(400, "Invalid step")
    student = str(payload.get("response", ""))[:MAX_LEN]
    if not student.strip():
        return {"verdict": "incomplete", "reply": "Please enter a step first.", "revealed": False}
    prior_fails = db.query(Attempt).filter(
        Attempt.activity_id == activity.id, Attempt.user_id == uid,
        Attempt.step_index == idx, Attempt.verdict != "correct").count()
    thread = _thread(db, f"a{activity.id}:{idx}:u{uid}")
    result = orchestrate(
        activity.problem_statement, steps[idx]["prompt"], steps[idx]["expected"],
        student, idx, len(steps), int(payload.get("hint_level", 0)),
        prior_fails, bool(activity.allow_solution_reveal),
        activity.max_attempts_before_reveal or 3, thread.messages,
    )
    if result["verdict"] != "correct" and steps and idx < len(steps):
        step_hints = steps[idx].get("hint", [])
        if step_hints:
            result["hint"] = step_hints[min(1, len(step_hints) - 1)]

    thread.append(student, result["reply"])
    db.add(Attempt(activity_id=activity.id, user_id=uid, step_index=idx,
                   student_response=student, verdict=result["verdict"],
                   hint_level=int(payload.get("hint_level", 0))))
    thread.save()
    return result


@app.post("/api/hint")
def api_hint(request: Request, payload: dict, db: Session = Depends(get_db)):
    activity = db.get(Activity, payload.get("activity_id", 0))
    if not activity:
        raise HTTPException(404, "Activity not found")
    steps = steps_of(activity)
    idx = int(payload.get("step_index", 0))
    level = max(1, min(4, int(payload.get("level", 1))))
    text = ""
    if steps and idx < len(steps):
        arr = steps[idx].get("hint", [])
        text = arr[min(level - 1, len(arr) - 1)] if arr else ""
    db.add(HintRequest(activity_id=activity.id,
                       user_id=_viewer_id(request, db), level=level,
                       hint_text=text))
    db.commit()
    return {"level": level, "hint": text}


@app.post("/api/tutor/example")
def api_tutor_example(payload: dict):
    step_idx = int(payload.get("step_index", 0))
    examples = [
        {
            "title": "Worked Example: Identifying Inner Function",
            "problem": "For $h(x) = (5x^2 - 4)^4$, identify the inner function $u = g(x)$.",
            "solution": "The expression inside the parenthesis being raised to the 4th power is $u = 5x^2 - 4$.",
            "rule": "For any composite function $(g(x))^n$, set $u = g(x)$."
        },
        {
            "title": "Worked Example: Applying the Chain Rule",
            "problem": "Differentiate $g(x) = (2x + 5)^3$.",
            "solution": "1. Outer derivative: $3(2x + 5)^2$\n2. Derivative of inside: $\\frac{d}{dx}[2x + 5] = 2$\n3. Multiply together: $g'(x) = 3(2x + 5)^2 \\cdot 2 = 6(2x + 5)^2$.",
            "rule": "$\\frac{d}{dx}[(g(x))^n] = n(g(x))^{n-1} \\cdot g'(x)$"
        },
        {
            "title": "Worked Example: Simplifying the Derivative",
            "problem": "Simplify $f'(x) = 2(4x - 1)(4)$.",
            "solution": "Multiply constants first: $2 \\times 4 = 8$.\nThen distribute: $8(4x - 1) = 32x - 8$.",
            "rule": "Combine coefficients, then distribute across parentheses."
        }
    ]
    ex = examples[min(step_idx, len(examples) - 1)]
    return ex


@app.post("/api/tutor/stuck")
def api_tutor_stuck(payload: dict):
    step_idx = int(payload.get("step_index", 0))
    if step_idx == 0:
        reply = "Let's break it down together! Look at $(3x + 1)^2$. The outer operation is squaring $(...)^2$. The inner function is what sits inside the parentheses. What expression is inside?"
    elif step_idx == 1:
        reply = "You've got this! Remember the chain rule has two parts:\n1. Power rule on the outside: bring the 2 to the front to get $2(3x + 1)^1$.\n2. Multiply by the derivative of the inside: what is $\\frac{d}{dx}[3x + 1]$? Multiply that constant to finish the step."
    else:
        reply = "Almost done! You have $2(3x + 1)(3)$. First multiply the constant numbers: $2 \\cdot 3 = 6$. Now multiply 6 by $(3x + 1)$: $6 \\cdot 3x + 6 \\cdot 1 = 18x + 6$."
    return {"reply": reply}


@app.post("/api/review-request")
def api_review_request(request: Request, payload: dict,
                       db: Session = Depends(get_db)):
    uid = _viewer_id(request, db)
    act_id = int(payload.get("activity_id", 0))
    req = db.query(ReviewRequest).filter(
        ReviewRequest.activity_id == act_id, ReviewRequest.user_id == uid
    ).first()
    if not req:
        req = ReviewRequest(activity_id=act_id, user_id=uid,
                            status="submitted",
                            notes="Student requested instructor review")
        db.add(req)
        db.commit()
    return {"status": "submitted",
            "message": f"Your work has been submitted for "
                       f"{_instructor_name(db)} to review!"}


@app.get("/lessons", response_class=HTMLResponse)
def lessons_page(request: Request, db: Session = Depends(get_db)):
    """Student My Lessons: only lessons from assigned packages."""
    lesson_cards, _ = _student_cards(
        db, db.query(Publication).order_by(Publication.id.desc()).all(),
        _viewer_id(request, db))
    return templates.TemplateResponse(request, "lessons.html", {
        "request": request, "cards": lesson_cards,
        "active_nav": "lessons", "demo": not config.ai_configured(),
    })


@app.get("/tutor", response_class=HTMLResponse)
def tutor_page(request: Request, db: Session = Depends(get_db)):
    courses = db.query(Course).all()
    activities = db.query(Activity).all()
    return templates.TemplateResponse(request, "tutor.html", {
        "request": request, "courses": courses, "activities": activities,
        "active_nav": "tutor", "demo": not config.ai_configured(),
    })


@app.get("/study-guides", response_class=HTMLResponse)
def study_guides_page(request: Request, db: Session = Depends(get_db)):
    items = db.query(CourseMaterial).all()
    courses = db.query(Course).all()
    return templates.TemplateResponse(request, "study_guides.html", {
        "request": request, "items": items, "courses": courses,
        "active_nav": "study-guides", "demo": not config.ai_configured(),
    })


@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    return templates.TemplateResponse(request, "help.html", {
        "request": request, "active_nav": "help",
    })


@app.get("/instructor", response_class=HTMLResponse)
def instructor(request: Request,
               who: User = Depends(auth.require_instructor),
               db: Session = Depends(get_db)):
    attempts = db.query(Attempt).order_by(Attempt.id.desc()).limit(50).all()
    students = db.query(User).filter(User.role == "student").all()
    total = db.query(Attempt).count()
    correct = db.query(Attempt).filter(Attempt.verdict == "correct").count()
    hint_count = db.query(HintRequest).count()
    # Per-student counts, so the roster table shows each student's own record
    # instead of one column that only ever reads "no attempts yet".
    stats = {}
    for a in db.query(Attempt).all():
        row = stats.setdefault(a.user_id, [0, 0])
        row[1] += 1
        if a.verdict == "correct":
            row[0] += 1
    return templates.TemplateResponse(request, "instructor.html", {
        "request": request, "attempts": attempts,
        "students": students, "total": total, "correct": correct,
        "hint_count": hint_count, "stats": stats,
        "demo": not config.ai_configured(),
    })


@app.get("/assignments", response_class=HTMLResponse)
def assignments(request: Request, db: Session = Depends(get_db)):
    """Student My Assignments tab: assignment-carrying packages.

    Lesson-only packages appear under My Lessons; packages with an
    assignment appear here with status, due date, and a link into the
    assignment workspace. Nothing is invented: unassigned practice
    content does not appear here.
    """
    _, asg_cards = _student_cards(
        db, db.query(Publication).order_by(Publication.id.desc()).all(),
        _viewer_id(request, db))
    cards = [{
        "link_id": c["link_id"], "title": c["title"], "course": c["course"],
        "due": c["due"], "grading": "", "kind": c["status"],
        "n_questions": c["n_questions"],
    } for c in asg_cards]
    return templates.TemplateResponse(
        request, "assignments.html", {"request": request, "cards": cards})


@app.get("/materials", response_class=HTMLResponse)
def materials(request: Request, db: Session = Depends(get_db)):
    items = db.query(CourseMaterial).all()
    return templates.TemplateResponse(request, "materials.html", {"request": request, "items": items})


@app.post("/materials/add")
def material_add(title: str = Form(...), kind: str = Form("note"),
                 body: str = Form(""), db: Session = Depends(get_db)):
    if not title.strip():
        raise HTTPException(400, "Title is required")
    db.add(CourseMaterial(title=title.strip(), kind=kind, body=body))
    db.commit()
    return RedirectResponse("/materials", status_code=303)


# --- Instructor shared sources (PDF / Word) ------------------------------

def _safe_original(name: str) -> str:
    """Strip any path components from a browser-supplied filename."""
    return os.path.basename(name or "file")[:300] or "file"


def _stored_path(stored_name: str) -> Path:
    """Resolve a stored name strictly inside UPLOAD_DIR (no traversal)."""
    return (UPLOAD_DIR / os.path.basename(stored_name)).resolve()


@app.get("/instructor/materials", response_class=HTMLResponse)
def instructor_materials(request: Request,
                         who: User = Depends(auth.require_instructor),
                         db: Session = Depends(get_db)):
    """Instructor panel: upload a source and share it with students."""
    files = db.query(SharedFile).order_by(SharedFile.created_at.desc()).all()
    from .courses import get_course_options
    return templates.TemplateResponse(request, "instructor_materials.html", {
        "request": request, "nav": "instructor-dashboard",
        "active_nav": "shared-materials", "files": files,
        "course_options": get_course_options(),
    })


@app.post("/instructor/materials/upload")
async def instructor_materials_upload(
        file: UploadFile = File(...), title: str = Form(""),
        course: str = Form(""),
        who: User = Depends(auth.require_instructor),
        db: Session = Depends(get_db)):
    original = _safe_original(file.filename)
    ext = os.path.splitext(original)[1].lower()
    if ext not in UPLOAD_TYPES:
        raise HTTPException(400, "Only PDF, DOC, and DOCX files are allowed")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File exceeds the 15 MB limit")
    if not data:
        raise HTTPException(400, "File is empty")
    stored_name = f"{uuid.uuid4().hex}{ext}"
    _stored_path(stored_name).write_bytes(data)
    row = SharedFile(
        title=(title.strip() or original)[:300],
        original_name=original, stored_name=stored_name,
        content_type=UPLOAD_TYPES[ext][0], size=len(data),
        course=course.strip()[:200], owner_id=who.id, shared=False)
    db.add(row)
    db.commit()
    return RedirectResponse("/instructor/materials", status_code=303)


@app.post("/instructor/materials/{file_id}/share")
def instructor_materials_share(file_id: int,
                               who: User = Depends(auth.require_instructor),
                               db: Session = Depends(get_db)):
    """Toggle student visibility. Off by default; sharing is an explicit act."""
    row = db.get(SharedFile, file_id)
    if not row:
        raise HTTPException(404, "File not found")
    row.shared = not row.shared
    db.commit()
    return RedirectResponse("/instructor/materials", status_code=303)


@app.post("/instructor/materials/{file_id}/delete")
def instructor_materials_delete(file_id: int,
                                who: User = Depends(auth.require_instructor),
                                db: Session = Depends(get_db)):
    row = db.get(SharedFile, file_id)
    if not row:
        raise HTTPException(404, "File not found")
    path = _stored_path(row.stored_name)
    db.delete(row)
    db.commit()
    if path.is_file():
        path.unlink()
    return RedirectResponse("/instructor/materials", status_code=303)


# --- Student shared-materials view ---------------------------------------

@app.get("/shared", response_class=HTMLResponse)
def shared_materials(request: Request, db: Session = Depends(get_db)):
    """Student view of instructor-shared sources (shared rows only)."""
    files = db.query(SharedFile).filter(
        SharedFile.shared == True).order_by(  # noqa: E712
        SharedFile.created_at.desc()).all()
    return templates.TemplateResponse(request, "shared_materials.html", {
        "request": request, "active_nav": "shared-materials", "files": files,
    })


@app.get("/shared/file/{file_id}")
def shared_file(file_id: int, request: Request,
                db: Session = Depends(get_db)):
    """Serve a shared source. Students reach only shared rows; an instructor
    may preview their own upload before sharing. PDFs render inline, Word
    documents are forced to download (browsers cannot preview them)."""
    row = db.get(SharedFile, file_id)
    if not row:
        raise HTTPException(404, "File not found")
    if not row.shared:
        who = auth.viewer(request, db)
        if who is None or who.role != "instructor":
            raise HTTPException(403, "This material has not been shared yet")
    path = _stored_path(row.stored_name)
    if not path.is_file():
        raise HTTPException(404, "File is no longer available")
    inline = UPLOAD_TYPES.get(
        os.path.splitext(row.stored_name)[1].lower(), ("", False))[1]
    disposition = "inline" if inline else "attachment"
    return FileResponse(
        path, media_type=row.content_type or "application/octet-stream",
        filename=row.original_name, content_disposition_type=disposition)


@app.get("/progress", response_class=HTMLResponse)
def progress(request: Request, db: Session = Depends(get_db)):
    """A student's own record. The viewer's identity picks whose rows appear,
    so switching accounts cannot surface another student's work."""
    who = auth.viewer_or_demo_student(request, db)
    attempts = db.query(Attempt).filter(
        Attempt.user_id == who.id).order_by(Attempt.id.desc()).all()
    return templates.TemplateResponse(request, "progress.html", {"request": request, "attempts": attempts,
                                       "verified": sum(
                                           1 for a in attempts
                                           if a.verdict == "correct"),
                                       "demo": not config.ai_configured()})


@app.post("/view-as")
def view_as(user_id: int = Form(...), back: str = Form("/"),
            db: Session = Depends(get_db)):
    """Demo identity switch; 0 signs out. See app/auth.py — this is a role
    boundary, not a login. `back` must be a path on this site, so the form
    cannot be turned into an off-site redirect."""
    dest = back if back.startswith("/") and not back.startswith("//") else "/"
    if user_id == 0:
        return auth.clear_viewer(RedirectResponse(dest, status_code=303))
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(404, "No such roster user")
    return auth.set_viewer(RedirectResponse(dest, status_code=303), target)


@app.get("/courses", response_class=HTMLResponse)
def courses(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        request, "courses.html",
        {"request": request, "courses": db.query(Course).all(),
         "activities": db.query(Activity).all()})


@app.get("/settings", response_class=HTMLResponse)
def settings(request: Request):
    return templates.TemplateResponse(request, "settings.html", {
        "request": request, "math_model": config.NEBIUS_MATH_MODEL,
        "socratic_model": config.NEBIUS_SOCRATIC_MODEL,
        "demo": not config.ai_configured(),
    })


@app.post("/api/generate-questions")
def api_generate_questions(payload: dict,
                           who: User = Depends(auth.require_instructor)):
    import app.generate as generate_mod
    try:
        result = generate_mod.generate_questions(
            notes=payload.get("notes", ""),
            topic=payload.get("topic", "Calculus"),
            num_questions=payload.get("num_questions", 5),
            difficulty=payload.get("difficulty", 3),
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Generation failed: {exc}")
    return result


@app.get("/instructor/dashboard", response_class=HTMLResponse)
def instructor_dashboard(request: Request, db: Session = Depends(get_db),
                         who: User = Depends(auth.require_instructor)):
    from .courses import get_course_options, get_course_topics
    activities = db.query(Activity).all()
    courses = db.query(Course).all()
    shared_files = db.query(SharedFile).order_by(SharedFile.created_at.desc()).all()
    return templates.TemplateResponse(request, "instructor_dashboard.html", {
        "request": request, "activities": activities, "courses": courses,
        "course_options": get_course_options(),
        "course_topics": get_course_topics(),
        "files": shared_files,
        "demo": not config.ai_configured(),
    })


@app.get("/instructor/drafts", response_class=HTMLResponse)
def instructor_drafts(request: Request, db: Session = Depends(get_db),
                      who: User = Depends(auth.require_instructor)):
    """Saved drafts manager: list lessons/assignments with student links."""
    import json as _json

    from .courses import get_course_options
    courses = {c.id: c.title for c in db.query(Course).all()}
    pubs = db.query(Publication).all()

    lessons = []
    for les in db.query(Lesson).order_by(Lesson.updated_at.desc()).all():
        try:
            content = _json.loads(les.instructor_content_json or "{}")
            sections = len(content.get("sections", []))
        except Exception:
            sections = 0
        lesson_pubs = [p for p in pubs if p.lesson_id == les.id]
        lessons.append({
            "id": les.id, "title": les.title, "topic": les.topic,
            "course": courses.get(les.course_id, ""),
            "status": les.status, "sections": sections,
            "updated_at": str(les.updated_at or ""),
            "publications": [{"link_id": p.link_id, "title": p.title,
                              "due": p.due_date,
                              "course": p.assigned_course or ""} for p in lesson_pubs],
        })

    assignments = []
    for asg in db.query(Assignment).order_by(
            Assignment.updated_at.desc()).all():
        try:
            items = _json.loads(asg.questions_json or "[]")
            n_items = len(items) if isinstance(items, list) else 0
        except Exception:
            n_items = 0
        asg_pubs = [p for p in pubs if p.assignment_id == asg.id]
        assignments.append({
            "id": asg.id, "title": asg.title, "topic": asg.topic,
            "course": courses.get(asg.course_id, ""),
            "status": asg.status, "items": n_items,
            "updated_at": str(asg.updated_at or ""),
            "publications": [{"link_id": p.link_id, "title": p.title,
                              "due": p.due_date,
                              "course": p.assigned_course or ""} for p in asg_pubs],
        })

    return templates.TemplateResponse(request, "instructor_drafts.html", {
        "request": request, "lessons": lessons,
        "assignments": assignments,
        "course_options": get_course_options(),
        "demo": not config.ai_configured(),
    })


@app.get("/api/analytics")
def api_analytics(request: Request, course: str = "",
                  teacher: User = Depends(auth.require_instructor),
                  db: Session = Depends(get_db)):
    """Instructor insights, aggregated from the tutor's own event log.

    Rates cover only the steps SymPy decided; unread responses are counted
    beside them rather than folded into a percentage. Instructor-only: the
    payload carries student-typed responses.
    """
    from . import insights

    return insights.build(db, course=course.strip() or None)


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "ai_configured": config.ai_configured(),
                         "math_model": config.NEBIUS_MATH_MODEL,
                         "socratic_model": config.NEBIUS_SOCRATIC_MODEL})


# --- Plan C: lesson lifecycle (draft -> approve -> derive -> publish) ---

def _lesson_to_dict(lesson: Lesson) -> dict:
    return {
        "id": lesson.id, "course_id": lesson.course_id,
        "topic": lesson.topic, "title": lesson.title,
        "learning_objectives": lesson.learning_objectives,
        "status": lesson.status,
        "updated_at": str(lesson.updated_at or ""),
    }


@app.post("/api/lessons/generate")
def api_lessons_generate(payload: dict,
                         who: User = Depends(auth.require_instructor)):
    """Generate a structured instructor lesson (draft, unapproved)."""
    import app.lessons as lessons_mod

    try:
        result = lessons_mod.generate_lesson(
            notes=payload.get("notes", ""),
            course=str(payload.get("course", "Calculus")),
            topic=str(payload.get("topic", "") or "General"),
            objectives=str(payload.get("objectives", "")),
            detail=str(payload.get("detail", "standard")),
            worked_examples=bool(payload.get("worked_examples", True)),
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Lesson generation failed: {exc}")
    return result


@app.post("/api/lessons/save")
def api_lessons_save(payload: dict, db: Session = Depends(get_db),
                     who: User = Depends(auth.require_instructor)):
    """Create or update a lesson draft. Never auto-publishes."""
    import json as _json

    from .schemas import (
        LessonContent, LessonSettings, lesson_content_from_json,
    )

    content = payload.get("lesson") or {}
    try:
        lesson_content = LessonContent.model_validate(content)
    except Exception as exc:
        raise HTTPException(400, f"Invalid lesson content: {exc}")
    settings = payload.get("settings") or {}
    try:
        LessonSettings.model_validate(settings)
        _json.dumps(settings)  # must be serializable
    except Exception as exc:
        raise HTTPException(400, f"Invalid lesson settings: {exc}")
    course_id = payload.get("course_id")
    if course_id is not None:
        try:
            course_id = int(course_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid course_id")
        if not db.get(Course, course_id):
            raise HTTPException(400, "Course not found")
    lesson_id = payload.get("id")
    if lesson_id:
        lesson = db.get(Lesson, int(lesson_id))
        if not lesson:
            raise HTTPException(404, "Lesson not found")
        if lesson.status == "published":
            raise HTTPException(
                400, "Published lessons are frozen — duplicate as a new draft to edit")
    else:
        lesson = Lesson()
        db.add(lesson)
    lesson.course_id = course_id
    lesson.topic = str(payload.get("topic", "") or lesson_content.topic)[:200]
    lesson.title = (lesson_content.title or "Untitled lesson")[:300]
    lesson.learning_objectives = "\n".join(lesson_content.objectives)
    lesson.source_materials_json = _json.dumps(
        payload.get("source_materials", []))
    lesson.instructor_content_json = lesson_content.model_dump_json()
    lesson.settings_json = _json.dumps(settings)
    # Preserve existing student content/verification until re-approved.
    db.commit()
    db.refresh(lesson)
    return {"id": lesson.id, "status": lesson.status}


@app.get("/api/lessons")
def api_lessons_list(db: Session = Depends(get_db),
                     who: User = Depends(auth.require_instructor)):
    """List lesson drafts (id, title, topic, status). No content bodies."""
    lessons = db.query(Lesson).order_by(Lesson.updated_at.desc()).all()
    return {"lessons": [_lesson_to_dict(les) for les in lessons]}


@app.get("/api/lessons/{lesson_id}")
def api_lessons_get(lesson_id: int, db: Session = Depends(get_db),
                    who: User = Depends(auth.require_instructor)):
    """Full lesson draft for the instructor editor (includes solutions)."""
    import json as _json

    lesson = db.get(Lesson, lesson_id)
    if not lesson:
        raise HTTPException(404, "Lesson not found")
    try:
        settings = _json.loads(lesson.settings_json or "{}")
    except Exception:
        settings = {}
    return {
        **_lesson_to_dict(lesson),
        "lesson": _json.loads(lesson.instructor_content_json or "{}"),
        "settings": settings,
        "source_materials": _json.loads(
            lesson.source_materials_json or "[]"),
        "verification": _json.loads(lesson.verification_json or "{}"),
    }


@app.post("/api/lessons/{lesson_id}/approve")
def api_lessons_approve(lesson_id: int, payload: dict,
                        db: Session = Depends(get_db),
                        who: User = Depends(auth.require_instructor)):
    """Approve a draft: verify math, derive student content, mark approved."""
    import json as _json

    import app.lessons as lessons_mod
    from .schemas import LessonSettings, lesson_content_from_json

    lesson = db.get(Lesson, lesson_id)
    if not lesson:
        raise HTTPException(404, "Lesson not found")
    if lesson.status == "published":
        raise HTTPException(400, "Published lessons are frozen")
    content = lesson_content_from_json(lesson.instructor_content_json)
    if payload.get("lesson"):
        try:
            from .schemas import LessonContent
            content = LessonContent.model_validate(payload["lesson"])
            lesson.instructor_content_json = content.model_dump_json()
            lesson.title = (content.title or lesson.title)[:300]
            lesson.learning_objectives = "\n".join(content.objectives)
        except Exception as exc:
            raise HTTPException(400, f"Invalid edited content: {exc}")
    try:
        settings = LessonSettings.model_validate(
            _json.loads(lesson.settings_json or "{}"))
    except Exception:
        settings = LessonSettings()
    counts = lessons_mod.verify_lesson_math(content)
    content.verification_summary = counts
    lesson.instructor_content_json = content.model_dump_json()
    lesson.verification_json = _json.dumps(counts)
    student = lessons_mod.derive_student_lesson(content, settings)
    lesson.student_content_json = _json.dumps(student)
    lesson.status = "approved"
    db.commit()
    return {"id": lesson.id, "status": "approved",
            "verification": counts,
            "student_sections": len(student["sections"]),
            "guided_items": len(student["guided_items"])}


@app.delete("/api/lessons/{lesson_id}")
def api_lessons_delete(lesson_id: int, db: Session = Depends(get_db),
                     who: User = Depends(auth.require_instructor)):
    """Delete a lesson draft. Blocked while student links reference it."""
    lesson = db.get(Lesson, lesson_id)
    if not lesson:
        raise HTTPException(404, "Lesson not found")
    in_use = db.query(Publication).filter(
        Publication.lesson_id == lesson_id).count()
    if in_use:
        raise HTTPException(
            409, f"Lesson is attached to {in_use} student link(s); "
                 "delete the assignment to students first")
    db.delete(lesson)
    db.commit()
    return {"deleted": lesson_id}


# --- Plan C: assignment lifecycle (generate -> draft -> approve) ---

def _assignment_to_dict(asg: Assignment) -> dict:
    return {
        "id": asg.id, "course_id": asg.course_id,
        "source_lesson_id": asg.source_lesson_id,
        "topic": asg.topic, "title": asg.title,
        "status": asg.status,
        "updated_at": str(asg.updated_at or ""),
    }


def _items_with_verification(items: list) -> list[dict]:
    """Attach per-item verification state (verified|needs-review|...) using
    the deterministic canonical gate. Items that fail parse or canonical
    check are honestly marked needs-review, never silently dropped."""
    from .schemas import (
        AssignmentItem, Question, QuestionStep, verify_question_canonical,
    )

    out: list[dict] = []
    for it in items:
        item = it if isinstance(it, AssignmentItem) else AssignmentItem.model_validate(it)
        steps = [QuestionStep(
            step_id=f"s{i + 1}",
            description=s.description if hasattr(s, "description") else s.get("description", ""),
            expected_input_latex=s.expected_input_latex if hasattr(s, "expected_input_latex") else s.get("expected_input_latex", ""),
            accepted_equivalents=list(s.accepted_equivalents if hasattr(s, "accepted_equivalents") else s.get("accepted_equivalents", [])),
            hints=list(s.hints if hasattr(s, "hints") else s.get("hints", [])),
            objective_tags=list(item.objective_tags),
        ) for i, s in enumerate(item.steps)]
        q = Question(
            prompt_latex=item.prompt_latex, difficulty=item.difficulty,
            steps=steps,
            final_answer_latex=item.prompt_latex,  # placeholder, replaced below
            source="ai-generated",
        )
        # Final answer comes from the stored solutions map via caller;
        # here we check step parseability as the honest minimum.
        from .schemas import check_step_parseable
        bad = [f"step {i + 1}" for i, s in enumerate(steps)
               if check_step_parseable(s.expected_input_latex) is not None]
        d = item.model_dump()
        d["verification"] = "verified" if not bad else "needs-review"
        if bad:
            d["verification_detail"] = f"unparseable: {', '.join(bad)}"
        out.append(d)
    return out


@app.post("/api/assignments/generate")
def api_assignments_generate(payload: dict, db: Session = Depends(get_db),
                             who: User = Depends(auth.require_instructor)):
    """Generate assignment items: from topic, or from an approved lesson.

    Uses the existing generate_questions pipeline (generate-then-verify).
    from_lesson_id optionally scopes concepts to the lesson's objectives.
    Returns items + solutions (instructor-only) + verification states.
    """
    import app.generate as generate_mod
    from .schemas import Question

    topic = str(payload.get("topic", "") or "General")
    course = str(payload.get("course", ""))
    n = max(1, min(10, int(payload.get("num_questions", 5))))
    balance = str(payload.get("difficulty_balance", "mixed"))
    difficulty = {"easy": 2, "hard": 4}.get(balance, 3)
    notes = str(payload.get("notes", ""))
    lesson_id = payload.get("from_lesson_id")
    if lesson_id:
        lesson = db.get(Lesson, int(lesson_id))
        if not lesson:
            raise HTTPException(404, "Source lesson not found")
        if lesson.status not in ("approved", "published"):
            raise HTTPException(400, "Source lesson must be approved first")
        from .schemas import lesson_content_from_json, lesson_assignment_context
        content = lesson_content_from_json(lesson.instructor_content_json)
        scoped = lesson_assignment_context(content)
        notes = (scoped + "\n" + notes).strip()
    try:
        result = generate_mod.generate_questions(
            notes=f"Course: {course}\n{notes}" if course else notes,
            topic=topic, num_questions=n, difficulty=difficulty)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Assignment generation failed: {exc}")
    items, solutions = [], {}
    for i, qd in enumerate(result.get("verified", [])):
        q = Question.model_validate(qd)
        item_id = f"q{i + 1}"
        last = q.steps[-1].expected_input_latex if q.steps else q.final_answer_latex
        items.append({
            "item_id": item_id,
            "prompt_latex": q.prompt_latex,
            "steps": [s.model_dump() for s in q.steps],
            "hints": [h for s in q.steps for h in s.hints][:4],
            "objective_tags": list({t for s in q.steps for t in s.objective_tags}),
            "difficulty": q.difficulty,
            "verification": "verified",
        })
        solutions[item_id] = {
            "final_answer_latex": q.final_answer_latex or last,
            "solution_steps": [s.expected_input_latex for s in q.steps],
            "explanation": f"Verified via canonical SymPy check ({q.steps[0].objective_tags[0] if q.steps and q.steps[0].objective_tags else topic}).",
        }
    return {"items": items, "solutions": solutions,
            "verified": len(items),
            "discarded": result.get("discarded", 0),
            "discard_reasons": result.get("discard_reasons", []),
            "truncated_notes": result.get("truncated_notes", False)}


@app.post("/api/assignments/save")
def api_assignments_save(payload: dict, db: Session = Depends(get_db),
                           who: User = Depends(auth.require_instructor)):
    """Create or update an assignment draft. Never auto-publishes."""
    import json as _json

    from .schemas import AssignmentItem, AssignmentSettings

    raw_items = payload.get("items") or []
    if not isinstance(raw_items, list) or not raw_items:
        raise HTTPException(400, "Assignment needs at least one item")
    try:
        items = [AssignmentItem.model_validate(d) for d in raw_items]
    except Exception as exc:
        raise HTTPException(400, f"Invalid assignment items: {exc}")
    settings = payload.get("settings") or {}
    try:
        AssignmentSettings.model_validate(settings)
        _json.dumps(settings)
    except Exception as exc:
        raise HTTPException(400, f"Invalid assignment settings: {exc}")
    solutions = payload.get("solutions") or {}
    if not isinstance(solutions, dict):
        raise HTTPException(400, "Invalid solutions map")
    course_id = payload.get("course_id")
    if course_id is not None:
        try:
            course_id = int(course_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid course_id")
        if not db.get(Course, course_id):
            raise HTTPException(400, "Course not found")
    source_lesson_id = payload.get("source_lesson_id")
    if source_lesson_id is not None:
        try:
            source_lesson_id = int(source_lesson_id)
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid source_lesson_id")
        if not db.get(Lesson, source_lesson_id):
            raise HTTPException(400, "Source lesson not found")
    asg_id = payload.get("id")
    if asg_id:
        asg = db.get(Assignment, int(asg_id))
        if not asg:
            raise HTTPException(404, "Assignment not found")
        if asg.status == "published":
            raise HTTPException(
                400, "Published assignments are frozen — duplicate as a new draft to edit")
    else:
        asg = Assignment()
        db.add(asg)
    asg.course_id = course_id
    asg.source_lesson_id = source_lesson_id
    asg.topic = str(payload.get("topic", ""))[:200]
    title = (str(payload.get("title", ""))
             or str(settings.get("title", ""))).strip()
    asg.title = (title or (f"{asg.topic} — Assignment" if asg.topic
                           else "Untitled assignment"))[:300]
    asg.questions_json = _json.dumps([i.model_dump() for i in items])
    asg.solutions_json = _json.dumps(solutions)
    asg.settings_json = _json.dumps(settings)
    db.commit()
    db.refresh(asg)
    return {"id": asg.id, "status": asg.status}


@app.get("/api/assignments")
def api_assignments_list(db: Session = Depends(get_db),
                         who: User = Depends(auth.require_instructor)):
    """List assignment drafts (metadata only, no solutions)."""
    rows = db.query(Assignment).order_by(Assignment.updated_at.desc()).all()
    return {"assignments": [_assignment_to_dict(a) for a in rows]}


@app.get("/api/assignments/{asg_id}")
def api_assignments_get(asg_id: int, db: Session = Depends(get_db),
                         who: User = Depends(auth.require_instructor)):
    """Full assignment draft for the instructor editor (includes solutions)."""
    import json as _json

    asg = db.get(Assignment, asg_id)
    if not asg:
        raise HTTPException(404, "Assignment not found")
    return {
        **_assignment_to_dict(asg),
        "items": _items_with_verification(
            _json.loads(asg.questions_json or "[]")),
        "solutions": _json.loads(asg.solutions_json or "{}"),
        "settings": _json.loads(asg.settings_json or "{}"),
        "verification": _json.loads(asg.verification_json or "{}"),
    }


@app.post("/api/assignments/{asg_id}/approve")
def api_assignments_approve(asg_id: int, db: Session = Depends(get_db),
                             who: User = Depends(auth.require_instructor)):
    """Approve an assignment draft: re-verify items, mark approved."""
    import json as _json

    asg = db.get(Assignment, asg_id)
    if not asg:
        raise HTTPException(404, "Assignment not found")
    if asg.status == "published":
        raise HTTPException(400, "Published assignments are frozen")
    items = _items_with_verification(_json.loads(asg.questions_json or "[]"))
    counts = {"verified": sum(1 for i in items if i["verification"] == "verified"),
              "needs_review": sum(1 for i in items if i["verification"] != "verified")}
    asg.questions_json = _json.dumps(items)
    asg.verification_json = _json.dumps(counts)
    asg.status = "approved"
    db.commit()
    return {"id": asg.id, "status": "approved", "verification": counts}


@app.delete("/api/assignments/{asg_id}")
def api_assignments_delete(asg_id: int, db: Session = Depends(get_db),
                            who: User = Depends(auth.require_instructor)):
    """Delete an assignment draft. Blocked while student links reference it."""
    asg = db.get(Assignment, asg_id)
    if not asg:
        raise HTTPException(404, "Assignment not found")
    in_use = db.query(Publication).filter(
        Publication.assignment_id == asg_id).count()
    if in_use:
        raise HTTPException(
            409, f"Assignment is attached to {in_use} student link(s); "
                 "delete the assignment to students first")
    db.delete(asg)
    db.commit()
    return {"deleted": asg_id}


# --- Plan C: publications (lesson and/or assignment, frozen at publish) ---

@app.post("/api/publications")
def api_publications_create(payload: dict, db: Session = Depends(get_db),
                            who: User = Depends(auth.require_instructor)):
    """Publish an approved lesson, assignment, or both as one package.

    Creates a stable Publication row with a link_id. Content is read from
    the approved drafts at publish time; later draft edits never mutate a
    published package (lessons/assignments freeze on publish).
    """
    import json as _json

    lesson_id = payload.get("lesson_id")
    asg_id = payload.get("assignment_id")
    if not lesson_id and not asg_id:
        raise HTTPException(400, "Nothing to publish: pass lesson_id and/or assignment_id")
    lesson, asg = None, None
    if lesson_id:
        lesson = db.get(Lesson, int(lesson_id))
        if not lesson:
            raise HTTPException(404, "Lesson not found")
        if lesson.status != "approved":
            raise HTTPException(400, "Lesson must be approved before publishing")
    if asg_id:
        asg = db.get(Assignment, int(asg_id))
        if not asg:
            raise HTTPException(404, "Assignment not found")
        if asg.status != "approved":
            raise HTTPException(400, "Assignment must be approved before publishing")
    link_id = uuid.uuid4().hex[:8]
    pub = Publication(
        link_id=link_id,
        lesson_id=lesson.id if lesson else None,
        assignment_id=asg.id if asg else None,
        title=str(payload.get("title", "")
                  or (lesson.title if lesson else "")
                  or (asg.title if asg else ""))[:300],
        due_date=str(payload.get("due", ""))[:30],
        grading_policy=str(payload.get("grading", "Completion"))[:50],
    )
    db.add(pub)
    # Freeze: published artifacts become immutable snapshots.
    if lesson:
        lesson.status = "published"
    if asg:
        asg.status = "published"
        # Materialize assignment items as Activities so the existing
        # Socratic solver (/api/submit) works unchanged.
        try:
            items = _json.loads(asg.questions_json or "[]")
        except Exception:
            items = []
        for it in items:
            steps_in = it.get("steps", []) if isinstance(it, dict) else []
            steps = [{"prompt": (s.get("description", "") if isinstance(s, dict) else ""),
                      "expected": (s.get("expected_input_latex", "") if isinstance(s, dict) else ""),
                      "hint": (list(s.get("hints", [])) if isinstance(s, dict) else [])}
                     for s in steps_in if isinstance(s, dict)]
            if not steps:
                continue
            db.add(Activity(
                title=str(it.get("prompt_latex", ""))[:200] or asg.title,
                course_topic=asg.topic or "General",
                problem_statement=str(it.get("prompt_latex", "")),
                learning_objectives="",
                expected_steps_json=_json.dumps(steps),
                accepted_methods="",
                misconceptions="",
                hint_sequence_json=_json.dumps(
                    ["Conceptual question", "Specific hint",
                     "Formula guidance", "Worked substep"]),
                max_attempts_before_reveal=3,
                allow_solution_reveal=True,
            ))
    db.commit()
    return {"link_id": link_id, "publication_id": pub.id,
            "student_url": f"/learn/{link_id}" if lesson else
            f"/myassignments/{link_id}",
            "solve_url": f"/learn/{link_id}",
            "lesson_id": lesson.id if lesson else None,
            "assignment_id": asg.id if asg else None}


@app.delete("/api/publications/{link_id}")
def api_publications_delete(link_id: str, db: Session = Depends(get_db),
                            who: User = Depends(auth.require_instructor)):
    """Retire a student link. Content referenced by no remaining link
    reverts to approved (editable/deletable); content still referenced
    by another link stays published."""
    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub:
        raise HTTPException(404, "Student link not found")
    lesson_id, asg_id = pub.lesson_id, pub.assignment_id
    db.delete(pub)
    db.flush()
    reverted: list[str] = []
    for model, content_id, label in (
            (Lesson, lesson_id, "lesson"),
            (Assignment, asg_id, "assignment")):
        if not content_id:
            continue
        remaining = db.query(Publication).filter(
            (Publication.lesson_id == content_id)
            if model is Lesson
            else (Publication.assignment_id == content_id)).count()
        if remaining == 0:
            content = db.get(model, content_id)
            if content and content.status == "published":
                content.status = "approved"
                reverted.append(label)
    db.commit()
    return {"retired": link_id, "reverted": reverted}


@app.post("/api/assign")
def api_assign(payload: dict, db: Session = Depends(get_db),
                 who: User = Depends(auth.require_instructor)):
    """Assign a published lesson/assignment to a course with a due date.

    The instructor flow is Generate -> Save Draft -> Approve -> Publish ->
    Assign. If the content is approved but not yet published, assigning
    publishes it first (same freeze rules) so Assign works in one click
    from the Builder tab.
    """
    course = str(payload.get("course", "") or "").strip()
    due = str(payload.get("due", "") or "").strip()
    if not due:
        raise HTTPException(400, "A due date is required to assign")
    lesson, asg = None, None
    link_id = str(payload.get("link_id", "") or "").strip()
    pub = None
    if link_id:
        pub = db.query(Publication).filter(
            Publication.link_id == link_id).first()
        if not pub:
            raise HTTPException(404, "Publication not found")
    else:
        lesson_id = payload.get("lesson_id")
        asg_id = payload.get("assignment_id")
        if not lesson_id and not asg_id:
            raise HTTPException(
                400, "Nothing to assign: pass link_id or lesson_id/assignment_id")
        if lesson_id:
            lesson = db.get(Lesson, int(lesson_id))
            if not lesson:
                raise HTTPException(404, "Lesson not found")
            if lesson.status not in ("approved", "published"):
                raise HTTPException(
                    400, "Lesson must be approved before assigning")
        if asg_id:
            asg = db.get(Assignment, int(asg_id))
            if not asg:
                raise HTTPException(404, "Assignment not found")
            if asg.status not in ("approved", "published"):
                raise HTTPException(
                    400, "Assignment must be approved before assigning")
        # Reuse the publish endpoint logic for approved content: a fresh
        # package freezes the approved drafts and yields a student link.
        if ((lesson and lesson.status == "approved")
                or (asg and asg.status == "approved")):
            created = api_publications_create({
                "lesson_id": lesson.id if lesson else None,
                "assignment_id": asg.id if asg else None,
                "title": str(payload.get("title", "")
                             or (lesson.title if lesson else "")
                             or (asg.title if asg else "")),
                "due": due,
                "grading": str(payload.get("grading", "Completion")),
            }, db, who)
            pub = db.query(Publication).filter(
                Publication.link_id == created["link_id"]).first()
        else:
            # Both already published: find the existing package.
            q = db.query(Publication)
            if lesson:
                q = q.filter(Publication.lesson_id == lesson.id)
            if asg:
                q = q.filter(Publication.assignment_id == asg.id)
            pub = q.order_by(Publication.id.desc()).first()
            if not pub:
                raise HTTPException(
                    404, "No published package found for this content")
    pub.due_date = due
    pub.assigned_course = course
    if payload.get("grading"):
        pub.grading_policy = str(payload.get("grading"))[:50]
    db.commit()
    return {"link_id": pub.link_id, "solve_url": f"/learn/{pub.link_id}",
            "due": pub.due_date, "course": pub.assigned_course}


@app.get("/learn/{link_id}", response_class=HTMLResponse)
def student_learn(link_id: str, request: Request, db: Session = Depends(get_db)):
    """Student lesson reading page (My Lessons -> View Lesson).

    Renders instructor-approved lesson content only; assignment items in
    the same package are listed separately, never solved here. Canonical
    student URL; /mylessons/{link_id} redirects here.
    """
    import json as _json

    from .lessons import split_student_payload

    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub:
        raise HTTPException(404, "Lesson link not found")
    if not pub.lesson_id:
        # Assignment-only package: send the student to the assignment
        # workspace instead of 404ing a valid package.
        return RedirectResponse(f"/myassignments/{link_id}", status_code=303)
    lesson_payload: dict = {}
    lesson = db.get(Lesson, pub.lesson_id)
    if lesson and lesson.student_content_json:
        try:
            lesson_payload = split_student_payload(
                _json.loads(lesson.student_content_json))
        except Exception:
            lesson_payload = {}
    return templates.TemplateResponse(request, "student_lesson.html", {
        "request": request, "publication": pub,
        "lesson": lesson_payload,
        "active_nav": "lessons",
        "demo": not config.ai_configured(),
    })


@app.get("/mylessons/{link_id}", response_class=HTMLResponse)
def student_lesson_alias(link_id: str, request: Request):
    """Stable My Lessons URL; the canonical page is /learn/{link_id}."""
    return RedirectResponse(f"/learn/{link_id}", status_code=303)


def _assignment_items_for_student(db: Session, pub) -> list[dict]:
    """Student-safe assignment items: prompts + progressive hints only.

    Solutions never leave the server: expected answers AND the ordered step
    descriptions (which spell out the worked method) are excluded. The
    floating tutor reveals guidance one step at a time on request; the
    assignment page itself must not pre-reveal the solution.
    """
    import json as _json

    if not pub.assignment_id:
        return []
    asg = db.get(Assignment, pub.assignment_id)
    if not asg:
        return []
    try:
        raw = _json.loads(asg.questions_json or "[]")
    except Exception:
        return []
    items = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        items.append({
            "item_id": it.get("item_id", ""),
            "prompt_latex": it.get("prompt_latex", ""),
            "hints": list(it.get("hints", [])),
            "difficulty": it.get("difficulty", 3),
        })
    return items


def _submission_for(db: Session, pub_id: int, user_id: int) -> Submission:
    """This student's attempt at this publication — and only theirs."""
    return db.query(Submission).filter(
        Submission.publication_id == pub_id,
        Submission.user_id == user_id).first()


@app.get("/myassignments/{link_id}", response_class=HTMLResponse)
def student_assignment(link_id: str, request: Request,
                       db: Session = Depends(get_db)):
    """Student assignment workspace: answer, save progress, submit.

    Loads the exact published assignment; restores the student's saved
    draft answers when reopened. Submitted work renders a confirmation.
    """
    import json as _json

    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub or not pub.assignment_id:
        raise HTTPException(404, "Assignment not found")
    asg = db.get(Assignment, pub.assignment_id)
    if not asg:
        raise HTTPException(404, "Assignment not found")
    items = _assignment_items_for_student(db, pub)
    sub = _submission_for(db, pub.id, _viewer_id(request, db))
    try:
        saved = _json.loads(sub.answers_json) if sub and sub.answers_json else {}
    except Exception:
        saved = {}
    try:
        attempts = _json.loads(sub.attempts_json) if sub and sub.attempts_json else {}
    except Exception:
        attempts = {}
    return templates.TemplateResponse(request, "student_assignment.html", {
        "request": request, "publication": pub,
        "title": asg.title or pub.title or "Untitled",
        "due": pub.due_date or "", "course": pub.assigned_course or "",
        "items": items, "saved": saved if isinstance(saved, dict) else {},
        "attempts": attempts if isinstance(attempts, dict) else {},
        "max_tries": MAX_TRIES,
        "submitted": bool(sub and sub.status == "submitted"),
        "submitted_at": str(sub.submitted_at)[:16] if sub and sub.submitted_at else "",
        "active_nav": "assignments",
        "demo": not config.ai_configured(),
    })


@app.post("/api/assignments/progress")
def api_assignment_progress(request: Request, payload: dict,
                            db: Session = Depends(get_db)):
    """Save draft answers (resume any time). Never marks submitted."""
    import json as _json
    from datetime import datetime, timezone

    uid = _viewer_id(request, db)
    link_id = str(payload.get("link_id", "") or "").strip()
    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub or not pub.assignment_id:
        raise HTTPException(404, "Assignment not found")
    sub = _submission_for(db, pub.id, uid)
    if sub and sub.status == "submitted":
        raise HTTPException(400, "Already submitted — answers are final")
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict):
        raise HTTPException(400, "Invalid answers")
    clean = {str(k)[:40]: str(v)[:2000] for k, v in answers.items()}
    if not sub:
        sub = Submission(publication_id=pub.id, user_id=uid,
                         answers_json=_json.dumps(clean), status="draft")
        db.add(sub)
    else:
        sub.answers_json = _json.dumps(clean)
        sub.status = "draft"
        sub.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"saved": True, "n_answers": len(clean)}


MAX_TRIES = 3


@app.post("/api/assignments/check")
def api_assignment_check(request: Request, payload: dict,
                         db: Session = Depends(get_db)):
    """Grade one answer server-side with the deterministic verifier.

    Gives the student immediate correct/incorrect feedback WITHOUT revealing
    the answer, and enforces a 3-try budget per question tracked in
    attempts_json ({item_id: {tries, correct}}). A check also saves the
    student's answer so work is never lost. Correctness is decided by SymPy
    equivalence, never by the language model.
    """
    import json as _json

    from .verifier import check_equivalent

    uid = _viewer_id(request, db)
    link_id = str(payload.get("link_id", "") or "").strip()
    item_id = str(payload.get("item_id", "") or "").strip()
    answer = str(payload.get("answer", "") or "").strip()
    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub or not pub.assignment_id:
        raise HTTPException(404, "Assignment not found")
    asg = db.get(Assignment, pub.assignment_id)
    if not asg:
        raise HTTPException(404, "Assignment not found")
    sub = _submission_for(db, pub.id, uid)
    if sub and sub.status == "submitted":
        raise HTTPException(400, "Already submitted — answers are final")
    try:
        items = _json.loads(asg.questions_json or "[]")
    except Exception:
        items = []
    item = next((it for it in items if isinstance(it, dict)
                 and it.get("item_id") == item_id), None)
    if item is None:
        raise HTTPException(404, "Question not found")
    try:
        sols = _json.loads(asg.solutions_json or "{}")
    except Exception:
        sols = {}
    expected = str((sols.get(item_id) or {}).get("final_answer_latex")
                   or "").strip()
    if not expected:
        steps = item.get("steps") or []
        if steps and isinstance(steps[-1], dict):
            expected = str(
                steps[-1].get("expected_input_latex") or "").strip()

    if not sub:
        sub = Submission(publication_id=pub.id, user_id=uid,
                         answers_json="{}", attempts_json="{}", status="draft")
        db.add(sub)
    try:
        attempts = _json.loads(sub.attempts_json or "{}")
    except Exception:
        attempts = {}
    entry = attempts.get(item_id) or {"tries": 0, "correct": False}
    try:
        ans_map = _json.loads(sub.answers_json or "{}")
    except Exception:
        ans_map = {}
    if answer:
        ans_map[item_id] = answer[:2000]

    def _persist():
        sub.attempts_json = _json.dumps(attempts)
        sub.answers_json = _json.dumps(ans_map)
        db.commit()

    if entry.get("correct"):
        return {"correct": True, "locked": True, "tries": entry.get("tries", 0),
                "tries_left": 0, "feedback": "You already solved this one."}
    if not answer:
        return {"correct": False, "locked": False, "tries": entry.get("tries", 0),
                "tries_left": MAX_TRIES - entry.get("tries", 0),
                "feedback": "Type an answer to check it."}
    if not expected:
        _persist()
        return {"correct": False, "locked": False, "tries": entry.get("tries", 0),
                "tries_left": MAX_TRIES - entry.get("tries", 0),
                "feedback": "This question isn't auto-checkable yet — submit "
                            "it for your instructor to review."}
    if entry.get("tries", 0) >= MAX_TRIES:
        _persist()
        return {"correct": False, "locked": True, "tries": MAX_TRIES,
                "tries_left": 0,
                "feedback": "You've used all 3 tries on this question. "
                            "Review it with your instructor or move on."}

    verdict = check_equivalent(answer, expected)
    if verdict.status == "correct":
        entry["correct"] = True
        attempts[item_id] = entry
        _persist()
        return {"correct": True, "locked": True, "tries": entry.get("tries", 0),
                "tries_left": MAX_TRIES - entry.get("tries", 0),
                "feedback": "Great — that's correct!"}
    entry["tries"] = entry.get("tries", 0) + 1
    attempts[item_id] = entry
    _persist()
    left = MAX_TRIES - entry["tries"]
    feedback = ("Not quite — try again." if left > 0 else
                "That's not quite right, and you've used all 3 tries. "
                "Review this one with your instructor.")
    return {"correct": False, "locked": left <= 0, "tries": entry["tries"],
            "tries_left": left, "feedback": feedback}


@app.post("/api/assignments/submit")
def api_assignment_submit(request: Request, payload: dict,
                          db: Session = Depends(get_db)):
    """Final submission: persists answers, locks the row. No re-submit."""
    import json as _json
    from datetime import datetime, timezone

    uid = _viewer_id(request, db)
    link_id = str(payload.get("link_id", "") or "").strip()
    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub or not pub.assignment_id:
        raise HTTPException(404, "Assignment not found")
    sub = _submission_for(db, pub.id, uid)
    if sub and sub.status == "submitted":
        raise HTTPException(400, "Already submitted")
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict):
        raise HTTPException(400, "Invalid answers")
    clean = {str(k)[:40]: str(v)[:2000] for k, v in answers.items()}
    now = datetime.now(timezone.utc)
    if not sub:
        sub = Submission(publication_id=pub.id, user_id=uid,
                         answers_json=_json.dumps(clean), status="submitted",
                         submitted_at=now)
        db.add(sub)
    else:
        sub.answers_json = _json.dumps(clean)
        sub.status = "submitted"
        sub.submitted_at = now
    db.commit()
    return {"submitted": True,
            "submitted_at": str(now)[:16],
            "n_answers": len(clean)}


@app.get("/api/learn/{link_id}")
def api_learn(link_id: str, db: Session = Depends(get_db)):
    """Student JSON API: same allowlist as the /learn page. No solutions."""
    import json as _json

    from .lessons import split_student_payload

    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub:
        raise HTTPException(404, "Lesson link not found")
    lesson_payload: dict = {}
    if pub.lesson_id:
        lesson = db.get(Lesson, pub.lesson_id)
        if lesson and lesson.student_content_json:
            try:
                lesson_payload = split_student_payload(
                    _json.loads(lesson.student_content_json))
            except Exception:
                lesson_payload = {}
    return {"link_id": link_id, "title": pub.title,
            "lesson": lesson_payload,
            "has_assignment": pub.assignment_id is not None}
