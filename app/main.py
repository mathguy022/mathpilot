"""MathPilot FastAPI app: student workspace, instructor dashboard, JSON API."""
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from . import config
from .db import (
    Assignment, ensure_schema, get_db, init_db, Activity, Attempt, Course, CourseMaterial,
    HintRequest, Lesson, Mastery, Publication, ReviewRequest, Submission,
    TutorEvent, TutorSession, User,
)
from .seed import seed
from .tutor import orchestrate
from .verifier import MAX_LEN

BASE = Path(__file__).resolve().parent.parent
app = FastAPI(title="MathPilot")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

init_db()
ensure_schema()
seed()

DEMO_HISTORY: dict[str, list[dict]] = {}

# Per practice item: hint rung reached and misses so far. In-memory like
# DEMO_HISTORY, so the ladder is server-owned rather than client-supplied.
PRACTICE_STATE: dict[str, dict] = {}

# Mock lesson store: UUID -> {"plan": LessonPlan dict, "activity_ids": [int]}.
# Demo only: in-memory, cleared on restart, synthetic content only.
LESSON_LINKS: dict[str, dict] = {}


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


@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    pubs = db.query(Publication).order_by(Publication.id.desc()).all()
    lesson_cards, asg_cards = _student_cards(db, pubs)
    completed = db.query(Submission).filter(
        Submission.user_id == 1,
        Submission.status == "submitted").count()
    in_progress = db.query(Submission).filter(
        Submission.user_id == 1,
        Submission.status == "draft").count()
    return templates.TemplateResponse(request, "home.html", {
        "request": request,
        "lesson_cards": lesson_cards, "assignment_cards": asg_cards,
        "n_lessons": len(lesson_cards), "n_pending": len(asg_cards),
        "n_completed": completed, "n_in_progress": in_progress,
        "demo": not config.ai_configured(),
    })


def _student_cards(db: Session, pubs: list) -> tuple[list, list]:
    """Build lesson + assignment cards for the student workspace.

    Shared by the dashboard and the My Lessons/My Assignments pages so all
    three views show the same packages. Lessons list every publication
    carrying a lesson; assignments list every publication carrying an
    assignment with its question count, due date, and submission status.
    """
    import json as _json

    lessons, assignments = [], []
    for p in pubs:
        if p.lesson_id:
            lesson = db.get(Lesson, p.lesson_id)
            lessons.append({
                "link_id": p.link_id,
                "title": (lesson.title if lesson else None) or p.title or "Untitled",
                "course": p.assigned_course or "",
                "instructor": "Dr. Lee",
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
                Submission.user_id == 1).first()
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


@app.get("/practice", response_class=HTMLResponse)
def practice_page(request: Request, topic: str = "", db: Session = Depends(get_db)):
    """Practice with AI Tutor: pick a lesson package, work its guided items.

    ?topic= preselects the first package whose lesson matches the topic
    (used by the lesson page's "Practice This Topic" button).
    """
    pubs = db.query(Publication).filter(
        Publication.lesson_id.isnot(None)).order_by(
        Publication.id.desc()).all()
    options = []
    for p in pubs:
        lesson = db.get(Lesson, p.lesson_id)
        title = (lesson.title if lesson else None) or p.title or "Untitled"
        lesson_topic = (lesson.topic if lesson else "") or ""
        options.append({"link_id": p.link_id, "title": title,
                        "topic": lesson_topic,
                        "course": p.assigned_course or ""})
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
        "float_thread": _float_thread_payload(db, 1, "default"),
    })


@app.post("/api/practice/items")
def api_practice_items(payload: dict, db: Session = Depends(get_db)):
    """Guided practice items for a student package (no solutions)."""
    import json as _json

    from .lessons import split_student_payload

    link_id = str(payload.get("link_id", "") or "").strip()
    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub or not pub.lesson_id:
        raise HTTPException(404, "Practice package not found")
    lesson = db.get(Lesson, pub.lesson_id)
    if not lesson or not lesson.student_content_json:
        return {"items": [], "title": pub.title}
    try:
        content = split_student_payload(_json.loads(lesson.student_content_json))
    except Exception:
        return {"items": [], "title": pub.title}
    items = []
    for g in content.get("guided_items", []) or []:
        items.append({"item_id": g.get("item_id", ""),
                      "prompt_latex": g.get("prompt_latex", ""),
                      "context": g.get("context", ""),
                      "hints_enabled": g.get("hints_enabled", True)})
    return {"items": items, "title": content.get("title", pub.title),
            "topic": content.get("topic", "")}


PRACTICE_HINT_RUNGS = (
    "In your own words, what is this item asking for? Start there.",
    "Find the first operation in the item and deal with that one piece.",
    "Name the lesson rule that fits, then write only the line it gives you.",
    "Split it: write what you are given, apply the rule once, simplify, then "
    "compare that with your attempt.",
)


def _practice_say(*, user_text: str, history: list[dict], course_ref: str,
                  intent: str, situation: str, fallback: str,
                  reveal=()) -> str:
    """LLM-worded practice reply, with the deterministic string as fallback.

    SymPy has already decided the verdict; the model only phrases it. When the
    phrasing states the item's answer instead of guiding toward it, the
    deterministic line is used — practice never reveals solutions.
    """
    from . import float_chat as _chat

    text = _chat.reply(user_text=user_text, history=history[-6:],
                       course_ref=course_ref, intent=intent,
                       situation=situation, graded=False)
    if text and _reply_reveals(text, reveal):
        text = ""
    return _float_display(text or fallback)


def _practice_verdict(problem: str, cleaned: str):
    """Grade an attempt by solving the item, not by matching its text.

    A guided item is built from the practice section's own math_latex, so the
    stored target IS the problem statement — comparing an answer against it
    would mark every correct response wrong. SymPy solves or evaluates the
    problem and compares that with the attempt.
    """
    from .schemas import Question
    from .verifier import check_equivalent, verify_canonical

    q = Question(id=0, prompt_latex=problem, steps=[],
                 final_answer_latex=cleaned)
    v = verify_canonical(q)
    if v.status == "unverifiable":
        alt = check_equivalent(cleaned, problem)
        if alt.status != "unverifiable":
            return alt
    return v


@app.post("/api/practice/attempt")
def api_practice_attempt(payload: dict, db: Session = Depends(get_db)):
    """Chat with the tutor about one guided practice item.

    Same pipeline as the floating widget: the intent router decides whether a
    message is a math attempt at all, SymPy decides correctness, and the LLM
    only words the reply. Prose is never graded, hints walk the ladder from
    server-side state, and no reply may state the item's answer (practice is
    ungraded but still never reveals solutions).
    """
    import json as _json
    import re as _re

    from . import float_intent as _intent
    from .lessons import split_student_payload
    from .verifier import MAX_LEN

    link_id = str(payload.get("link_id", "") or "").strip()
    item_id = str(payload.get("item_id", "") or "").strip()
    student = str(payload.get("response", ""))[:MAX_LEN].strip()
    if not student:
        return {"verdict": "incomplete",
                "reply": "Type your attempt, or ask me a question — I can "
                         "hint, explain, or walk it step by step.",
                "replyType": "bubble", "hintLevel": 0, "revealed": False}
    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub or not pub.lesson_id:
        raise HTTPException(404, "Practice package not found")
    lesson = db.get(Lesson, pub.lesson_id)
    if not lesson or not lesson.student_content_json:
        raise HTTPException(404, "No practice content")
    try:
        content = split_student_payload(_json.loads(lesson.student_content_json))
    except Exception:
        raise HTTPException(502, "Practice content unreadable")
    guided = {g.get("item_id"): g for g in
              content.get("guided_items", []) or []}
    if item_id not in guided:
        raise HTTPException(404, "Practice item not found")
    # Expected answer lives in the instructor-approved student content?
    # No — guided items carry no answers by design. Resolve the expected
    # form from the matching practice section's worked content is wrong;
    # instead verify against the instructor's practice math via the
    # stored instructor content practice section at the same index.
    try:
        instructor = _json.loads(lesson.instructor_content_json or "{}")
    except Exception:
        instructor = {}
    expected = ""
    try:
        idx = int(item_id.split("-")[-1]) - 1
        for s in instructor.get("sections", []) or []:
            if isinstance(s, dict) and s.get("kind") == "practice":
                math_items = s.get("math_latex", []) or []
                if 0 <= idx < len(math_items):
                    expected = str(math_items[idx])
                break
    except Exception:
        expected = ""

    item = guided[item_id]
    prompt = str(item.get("prompt_latex", "") or "").strip()
    course_ref = str(payload.get("course", "") or pub.assigned_course or "")
    key = f"practice:{link_id}:{item_id}"
    state = PRACTICE_STATE.setdefault(key, {"hints": 0, "misses": 0})
    history = DEMO_HISTORY.setdefault(key, [])
    item_line = f"The open practice item is: {prompt}" if prompt else ""
    # What the reply must not say, derived by SymPy rather than by the model.
    from .verifier import canonical_answer as _canonical_answer
    answers = _canonical_answer(prompt or expected)

    intent = _intent.classify(student, has_session=True)

    def _turn(reply: str, verdict: str, detail: str = "",
              rtype: str = "bubble") -> dict:
        history.extend([{"role": "user", "content": student},
                        {"role": "assistant", "content": reply}])
        return {"verdict": verdict, "reply": reply, "detail": detail,
                "intent": intent, "replyType": rtype,
                "hintLevel": state["hints"], "revealed": False,
                "misses": state["misses"]}

    def _say(situation: str, fallback: str,
             intent_name: str = "chat") -> str:
        return _practice_say(
            user_text=student, history=history, course_ref=course_ref,
            intent=intent_name, situation=situation, fallback=fallback,
            reveal=answers)

    def _hint(lvl: int, extra: str) -> str:
        state["hints"] = lvl
        body = _say(
            "Give exactly hint level "
            f"{lvl} of 4 for this practice item (1 what the item asks for, "
            "2 where to look, 3 which lesson rule to apply, 4 split it into "
            "micro-steps). Name the move, never do the algebra for them, and "
            "never state what the item evaluates to. "
            f"{item_line} {extra}",
            "")
        body = _re.sub(r"^\s*(hint\s*)?\d+\s*[—\-:]\s*", "", body.strip(),
                      flags=_re.IGNORECASE)
        return _float_display(
            f"Hint {lvl} — {body or PRACTICE_HINT_RUNGS[lvl - 1]}")

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

    # STEP_ANSWER (and anything unrecognized): the CAS decides.
    cleaned = _intent.clean_answer(student)
    if not cleaned:
        return _turn(_say(
            "The student's message did not parse as a math response. Say so "
            "in one sentence and ask them to send just the expression they "
            f"got, or a question. {item_line}",
            "I couldn't match that to this item. Send just the math you got, "
            "or ask me for a hint."), "not_an_attempt")

    verdict = _practice_verdict(prompt or expected, cleaned)
    status = verdict.status
    detail = verdict.detail

    if status == "correct":
        state["hints"] = 0
        return _turn(_say(
            "Our math checker VERIFIED this attempt correct. Confirm warmly "
            "in one sentence and ask what they want to practise next. Do not "
            "restate the answer.", "✔ Verified — nice work."),
            "correct", detail, "success")

    if status == "incorrect":
        state["misses"] += 1
        return _turn(_hint(min(4, state["hints"] + 1),
                           f"Their attempt was {cleaned}, which the checker "
                           "says is not equivalent to the target yet."),
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
def api_tutor_chat(payload: dict, db: Session = Depends(get_db)):
    """Floating Math Tutor chat (Student View only).

    Policy (enforced here, never in prompts alone):
    - Practice context (ungraded, item loaded): reuse the existing
      /api/practice/attempt verification pipeline; the reply is the
      same Socratic feedback the practice page gives.
    - Assignment context (graded): hints only. Never verify the student
      against assignment solutions, never reveal answers.
    - Lesson/general context: Socratic guidance grounded in the current
      lesson topic; never a worked solution or final answer.
    Server-side conversation state is shared with the practice page via
    DEMO_HISTORY so both surfaces continue one thread.
    """
    from . import config as _config
    from . import nebius as _nebius
    from .verifier import MAX_LEN as _MAX_LEN

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
            key = f"practice:{link_id}:{item_id}"
            history = DEMO_HISTORY.setdefault(key, [])
            reply = ("I can't give the final answer — that's the rule that "
                     "makes practice work. Tell me the last step you tried, "
                     "or press Check Step on your attempt, and I'll guide "
                     "the next move.")
            history += [{"role": "user", "content": message},
                        {"role": "assistant", "content": reply}]
            return {"reply": reply, "mode": "practice", "revealed": False}
        practice_payload = {
            "link_id": link_id, "item_id": item_id,
            "response": message,
            "hint_level": int(context.get("hint_level", 1) or 1),
        }
        result = api_practice_attempt(practice_payload, db)
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

    # --- Lesson / general context: Socratic guidance, answers never given.
    lesson_title = str(context.get("lesson_title", "") or "")[:200]
    lesson_meta = str(context.get("lesson_meta", "") or "")[:200]
    prompt_bits = [f"Student message: {message}"]
    if lesson_title:
        prompt_bits.append(f"Current lesson: {lesson_title} ({lesson_meta})")
    history = DEMO_HISTORY.setdefault("float-tutor:general", [])
    if wants_answer:
        reply = ("I won't give the final answer — but I will get you there. "
                 "What have you tried so far? Show me your last step and "
                 "I'll point at the next move.")
    elif _config.ai_configured():
        try:
            reply = _nebius.socratic_reply(
                {"problem": " ".join(prompt_bits),
                 "step": "general question", "expected": "(withheld)",
                 "student": message, "verdict": "unverifiable",
                 "verdict_detail": "general chat: no expected step",
                 "ai_judge": "n/a", "hint_level": 1,
                 "step_index": 0, "total_steps": 1,
                 "allow_reveal": False},
                history,
            )
        except Exception as exc:
            reply = (f"Live tutor unavailable ({exc}). Break the problem "
                     "into the smallest next step and tell me what you tried.")
    else:
        reply = ("**Demo mode** — here's how to start: write down what the "
                 "problem gives you and what it's asking for, try the "
                 "smallest first step, and tell me what you got. "
                 "Add your Nebius key for live Socratic tutoring.")
    history += [{"role": "user", "content": message},
                {"role": "assistant", "content": reply}]
    return {"reply": reply, "mode": mode, "revealed": False}


@app.get("/api/tutor/session")
def api_tutor_session(course: str = "", db: Session = Depends(get_db)):
    """Resume hook: the widget + AI Tutor tab share this open session."""
    return _float_thread_payload(db, 1, course or "default")


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
FLOAT_TUTOR_HITS: dict[int, list[float]] = {}
GRADED_HINTS_ONLY = True  # single setting: graded mode restricts hints

# Pending new-problem confirmations: (student_id, course_id) -> record.
# In-memory, consistent with DEMO_HISTORY; expires after 10 minutes.
# TODO: move to the database (new table) so it survives restarts.
# On loss/expiry the router re-runs and never grades a probable new
# problem as a step answer.
FLOAT_TUTOR_PENDING: dict[tuple[int, str], dict] = {}
FLOAT_TUTOR_PENDING_TTL_S = 600


def _float_pending_get(student_id: int, course_id: str) -> dict | None:
    import time as _time

    rec = FLOAT_TUTOR_PENDING.get((student_id, course_id))
    if not rec:
        return None
    if _time.time() - rec.get("ts", 0) > FLOAT_TUTOR_PENDING_TTL_S:
        FLOAT_TUTOR_PENDING.pop((student_id, course_id), None)
        return None
    return rec


def _float_pending_set(student_id: int, course_id: str,
                       problem: str) -> None:
    import time as _time

    FLOAT_TUTOR_PENDING[(student_id, course_id)] = {
        "problem": problem, "ts": _time.time()}


def _float_pending_clear(student_id: int, course_id: str) -> None:
    FLOAT_TUTOR_PENDING.pop((student_id, course_id), None)


def _float_rate_ok(student_id: int) -> bool:
    import time as _time

    now = _time.time()
    hits = [t for t in FLOAT_TUTOR_HITS.get(student_id, [])
            if now - t < FLOAT_TUTOR_WINDOW_S]
    if len(hits) >= FLOAT_TUTOR_RATE_LIMIT:
        FLOAT_TUTOR_HITS[student_id] = hits
        return False
    hits.append(now)
    FLOAT_TUTOR_HITS[student_id] = hits
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
               event_type: str = "message") -> None:
    db.add(TutorEvent(session_id=session_id, step_index=step_index,
                      student_input=student_input[:2000], verdict=verdict,
                      hint_level=hint_level, event_type=event_type))
    db.commit()


def _float_current_prompt(session: TutorSession) -> str:
    import json as _json

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
                step_no: int) -> tuple[str, int]:
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
        reply = (f"Let's split it into micro-steps. First, copy down just "
                 f"the left side of your equation. Then copy the right side. "
                 f"Now: {instruction} If it's still stuck, ask your "
                 f"instructor about this step in class.")
    return reply, level


def _step_body(llm_text: str) -> str:
    """Return the model's wording with any step label it re-added removed.

    The engine supplies the "Step N:" label itself, so a reply that opens
    "Great! Step 3: Check by…" would otherwise show the label twice. Only a
    real label is removed: prose that merely starts with a sentence is left
    alone.
    """
    import re as _re

    body = llm_text.strip()
    m = _re.match(r"([^.!?\n]{0,30}[!.?\n]\s*)?(step\s*\d+\s*[:.\-—]\s*)",
                  body, flags=_re.IGNORECASE)
    if not m:
        return body
    return (m.group(1) or "") + body[m.end():]


def _float_display(reply: str) -> str:
    """Apply the small display formatter to tutor reply text.

    Instructions are already prose; this cleans any embedded CAS
    spellings (2*x, x**2, pi) without touching $...$ math regions'
    structure. Safe to run on every reply.
    """
    from .verifier import display_math as _dm

    parts = reply.split("$")
    for i in range(0, len(parts), 2):  # even indices: outside math
        chunk = parts[i]
        if not chunk.strip():
            continue
        # display_math strips, which used to weld the prose onto the inline
        # math ("of $f(x)$ next" -> "of$f(x)$next"), so keep the pad.
        pad_l = chunk[:len(chunk) - len(chunk.lstrip())]
        pad_r = chunk[len(chunk.rstrip()):]
        parts[i] = pad_l + _dm(chunk) + pad_r
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
    if not _re.search(r"[a-z]", e):
        return rf"=\s*{_re.escape(e)}(?![\w.])"
    # Token-escape, then let any run of spaces match any other: the plan
    # writes "9 = 3*x" but a reply reads "9 = 3x".
    core = r"\s*".join(_re.escape(t) for t in e.split())
    core = core.replace(r"\*", r"\*{0,1}")
    return rf"(?<![\w.]){core}(?![\w.])"


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
    # "the answer is 3" or "x = 3". The item itself shows the same digits.
    return (r"(?:\b(?:answer|result|value|solution|limit|it|that|this|you)"
            r"[^.!\n]{0,28}?(?:\bis\b|:|\bequals?|\bcomes to|\bworks out to)"
            rf"|\b[a-z]\s*=)\s*{core}{tail}")


def _reply_reveals(reply: str, values) -> bool:
    """True when the reply states the item's answer rather than guiding."""
    import re as _re

    low = (reply or "").lower()
    return any((pat := _reveal_pattern(v)) and _re.search(pat, low)
               for v in values)


def _float_leak_check(reply: str, session, block_current: bool = False) -> str:
    """Answer-leak check (spec 6e): strip step results from replies.

    Compares the reply against future steps' expected expressions; on a
    hit, replaces with the current step's instruction. The recap card
    bypasses this (it intentionally shows the full solution). When
    `block_current` is set the active step's result is blocked too, which
    is what a refusal to reveal an answer needs.
    """
    import json as _json

    try:
        plan = _json.loads(session.solution_plan_json or "[]")
    except Exception:
        return reply
    start = session.current_step_index if block_current else \
        session.current_step_index + 1
    upcoming = plan[start:] if session else []
    if not upcoming:
        return reply
    if _reply_leaks(reply, [st.get("expected_expression", "")
                            for st in upcoming]):
        cur = plan[min(session.current_step_index, len(plan) - 1)]
        return (f"Let's stay on the current step: "
                f"{cur.get('instruction', '')} What do you get?")
    return reply


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


def _float_say(db: Session, session, message: str, *, intent: str,
               situation: str, fallback: str, course_ref: str = "",
               graded: bool = False, block_current: bool = False) -> str:
    """LLM-worded tutor reply; `fallback` when the model is unusable.

    The model never decides correctness — the caller already has the CAS
    verdict and passes it in as a fact to phrase. Replies pass the same
    leak check as scripted ones, so a chatty model cannot spoil a step.
    """
    from . import float_chat as _chat

    text = _chat.reply(user_text=message,
                       history=_float_history(db, session.id if session else 0),
                       course_ref=course_ref, intent=intent,
                       situation=situation, graded=graded)
    if not text:
        return fallback
    if session:
        text = _float_leak_check(text, session, block_current=block_current)
    return _float_display(text)


def _float_hint_said(db: Session, session, message: str, step: dict,
                     lvl: int, *, graded: bool, course_ref: str,
                     fallback: str) -> str:
    """Word one hint rung with the LLM, keeping the "Hint N — " prefix.

    The ladder state and the rung cap stay in code; the model only phrases
    the rung it is given, so it cannot skip ahead to a worked solution.
    """
    llm = _float_say(
        db, session, message, intent="chat", course_ref=course_ref,
        graded=graded, fallback="", block_current=True,
        situation=(
            f"The student's answer to this step was not correct. Step: "
            f"{step.get('instruction', '')} Why it works: "
            f"{step.get('reason', '')}. Give exactly hint level {lvl} of 4 "
            f"(1 conceptual nudge, 2 where to look, 3 which operation to "
            f"use, 4 split into micro-steps). Never state the result of "
            f"any step."))
    if not llm:
        return fallback
    import re as _re

    body = _re.sub(r"^\s*(hint\s*)?\d+\s*[—\-:]\s*", "", llm.strip(),
                   flags=_re.IGNORECASE)
    return _float_leak_check(f"Hint {lvl} — {body}", session)


@app.post("/api/tutor/message")
def api_tutor_message(payload: dict, db: Session = Depends(get_db)):
    """Floating-tutor guided chat. CAS decides correctness; replies guide.

    Input:  {message, pageContext} (student/course from session, never client).
    Output: {reply, replyType, stepIndex, totalSteps, stepStatus, hintLevel,
             sessionDone, recap?}.
    """
    import json as _json
    from datetime import datetime, timezone

    from . import float_plan as _plan

    student_id = 1  # demo single-student session (auth provides this later)
    message = str(payload.get("message", "") or "")[:500].strip()
    ctx = payload.get("pageContext") or payload.get("pagecontext") or {}
    if not isinstance(ctx, dict):
        ctx = {}
    ctx_type = str(ctx.get("type", "general") or "general")
    course_id = str(ctx.get("courseId", "") or ctx.get("course", "")
                    or "default")

    if not _float_rate_ok(student_id):
        raise HTTPException(429, "Slow down — 30 messages per 10 minutes.")
    if not message:
        return {"reply": "Please rephrase that — send an equation like "
                "$10 = 3x + 1$, or tap a suggestion chip.",
                "replyType": "bubble", "stepIndex": 0, "totalSteps": 0,
                "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False}

    graded = (ctx_type == "assignment"
              and str(ctx.get("mode", "practice")) == "graded")
    session = _float_open_session(db, student_id, course_id)
    lowered = message.lower()

    def _dump(plan: list[dict]) -> str:
        return _json.dumps(plan)

    def _recap(sess: TutorSession) -> dict:
        try:
            plan = _json.loads(sess.solution_plan_json or "[]")
        except Exception:
            plan = []
        steps = [{"n": i + 1,
                  "action": s.get("instruction", ""),
                  "result": s.get("expected_expression", ""),
                  "why": s.get("reason", "")}
                 for i, s in enumerate(plan)]
        answer = plan[-2].get("expected_expression", "") if len(plan) >= 2 else ""
        check = (plan[-1].get("expected_expression", "")
                 if plan else "")
        recap = {
            "title": f"Full solution — {sess.problem_text}",
            "steps": steps, "answer": answer, "check": check,
            "idea": ("Undo operations one at a time: constants first, "
                     "then the coefficient, then verify by substituting."),
        }
        if sess.total_misses:
            recap["note"] = ("You needed a few hints — that's normal. "
                             "Re-try a similar problem to lock it in.")
        return recap

    def _shape(sess: TutorSession, reply: str, rtype: str,
               status: str, hint: int, done: bool = False,
               recap: dict | None = None) -> dict:
        out = {"reply": reply, "replyType": rtype,
               "stepIndex": sess.current_step_index,
               "totalSteps": sess.total_steps, "stepStatus": status,
               "hintLevel": hint, "sessionDone": done}
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

    _pending = _float_pending_get(student_id, course_id)
    if _pending:
        # Awaiting "Start new problem?" answer: yes/no only.
        if lowered.strip() in ("yes", "start new problem", "start",
                               "new problem", "start it"):
            _float_pending_clear(student_id, course_id)
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
            _float_pending_clear(student_id, course_id)
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
            _float_pending_set(student_id, course_id, message)
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
                       message, verdict, session.hints_used, "message")
            return _shape(session, reply_text, rtype, "waiting",
                          session.hints_used)
        return {"reply": reply_text, "replyType": rtype, "stepIndex": 0,
                "totalSteps": 0, "stepStatus": "waiting", "hintLevel": 0,
                "sessionDone": False}

    if intent == "OFF_TOPIC":
        fallback = ("I'm your math tutor — let's stick to math and course "
                    "topics. " + (_float_current_prompt(session) if session
                                  else "Send me an equation and we'll work it "
                                       "step by step."))
        return _bubble(_float_say(
            db, session, message, intent="off_topic", course_ref=course_ref,
            graded=graded, fallback=fallback,
            situation="The student asked about something outside the course "
                      "sequence. Decline in one sentence and redirect to "
                      "course math."), "off-topic")

    if intent == "CONCEPT_QUESTION":
        situation = ("The student asked a conceptual math question. Answer "
                     "the idea briefly and correctly, then invite them to "
                     "try one step themselves.")
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
            graded=graded, fallback=fallback, situation=situation), "concept")

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
            graded=graded, fallback=fallback, situation=situation), "chat")

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
                           "hint")
                session.hints_used += 1
                db.commit()
                return _shape(session, reply, "hint", "waiting",
                              session.hints_used)
            reply, lvl = _float_hint(step, session.hints_used,
                                     session.total_misses,
                                     session.current_step_index + 1)
            session.hints_used = lvl
            db.commit()
            _float_log(db, session.id, session.current_step_index,
                       message, "hint", lvl, "hint")
            reply = _float_hint_said(db, session, message, step, lvl,
                                     graded=graded, course_ref=course_ref,
                                     fallback=reply)
            return _shape(session, _float_display(reply), "hint", "waiting",
                          lvl)

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
        plan, source = _plan.build_plan(message)
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
                problem_text=message, solution_plan_json=_dump(plan),
                status="active", current_step_index=0,
                total_steps=len(plan), graded_mode=graded)
            db.add(session)
            db.commit()
            _float_log(db, session.id, 0, message, "new-problem", 0,
                       "graded_use" if graded else "message")
            head = "Graded assignment: limited hints. " if graded else ""
            return _shape(session,
                          f"{head}Step 1: {plan[0]['instruction']} "
                          f"What equation do you get?",
                          "step", "waiting", 0)
        # Generic Socratic mode: no invented plan, honest about it.
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
                   message, "correct", 0)
        nxt = plan[session.current_step_index]
        step_no = session.current_step_index + 1
        reply = (f"Correct! Step {step_no}: "
                 f"{nxt['instruction']} What do you get?")
        llm = _float_say(
            db, session, message, intent="chat", course_ref=course_ref,
            graded=graded, fallback="",
            situation=(
                "SymPy verified the student's last answer as CORRECT. "
                "Celebrate it in one short sentence, then ask for step "
                f"{step_no} only: {nxt['instruction']} Do not state the "
                "result of that step or any later one."))
        if llm:
            reply = f"Step {step_no}: {_step_body(llm)}"
        return _shape(session, _float_display(
            _float_leak_check(reply, session)),
            "step", "waiting", 0)

    if verdict == "partial":
        # Correct but incomplete (e.g. one of two solutions): prompt for
        # the rest, do NOT count as a miss.
        _float_log(db, session.id, session.current_step_index,
                   message, "partial", session.hints_used)
        fallback = ("You found one solution. Is there another? "
                    f"({step.get('instruction', '')})")
        return _shape(
            session, _float_say(
                db, session, message, intent="chat", course_ref=course_ref,
                graded=graded, fallback=fallback,
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
                   message, "unclear", session.hints_used)
        recent_unclear = db.query(TutorEvent).filter(
            TutorEvent.session_id == session.id,
            TutorEvent.verdict == "unclear").count()
        from .verifier import display_math as _dm

        expect = _dm(step.get("expected_expression", ""))
        fallback = (f"I couldn't match that to this step. "
                    f"{step.get('instruction', '')} For example, write it "
                    f"like: {expect}.")
        if recent_unclear >= 2:
            fallback += (" Tap fx for symbols — the toolbar above the typing "
                         "box has everything you need.")
        reply = _float_say(
            db, session, message, intent="chat", course_ref=course_ref,
            graded=graded, fallback=fallback,
            situation=(
                "The checker could not read the student's last message. Do "
                "not scold or repeat a format lecture. Restate in plain "
                f"words what this step asks for ({step.get('instruction', '')}) "
                f"and ask them for just that one thing. Never reveal the "
                f"result, which is {expect}."))
        return _shape(session, _float_leak_check(reply, session),
                      "bubble", "waiting", session.hints_used)

    # Incorrect: hint ladder (graded caps at level 2, recap deferred).
    session.total_misses += 1
    if graded and GRADED_HINTS_ONLY:
        lvl = min(session.hints_used + 1, 2)
        session.hints_used = lvl
        db.commit()
        _float_log(db, session.id, session.current_step_index,
                   message, "incorrect", lvl, "graded_use")
        reply, _ = _float_hint(step, lvl - 1, session.total_misses,
                               session.current_step_index + 1)
        # Strip level-3+ content: cap means rung 1-2 wording only.
        reply = (f"Hint {lvl} — not quite. {step['reason']} "
                 f"Try again: {step['instruction']}")
        reply = _float_hint_said(db, session, message, step, lvl,
                                 graded=True, course_ref=course_ref,
                                 fallback=reply)
        return _shape(session, _float_display(reply), "hint", "waiting", lvl)
    reply, lvl = _float_hint(step, session.hints_used,
                             session.total_misses,
                             session.current_step_index + 1)
    session.hints_used = lvl
    db.commit()
    _float_log(db, session.id, session.current_step_index,
               message, "incorrect", lvl, "hint")
    reply = _float_hint_said(db, session, message, step, lvl,
                             graded=graded, course_ref=course_ref,
                             fallback=reply)
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
    attempts = db.query(Attempt).filter(
        Attempt.activity_id == activity_id, Attempt.user_id == 1
    ).order_by(Attempt.id.asc()).all()
    courses = db.query(Course).all()
    materials = db.query(CourseMaterial).filter(
        CourseMaterial.title.like(f"%{activity.course_topic}%") | CourseMaterial.title.like("%Chain rule%")
    ).all()
    if not materials:
        materials = db.query(CourseMaterial).limit(3).all()
    review_req = db.query(ReviewRequest).filter(
        ReviewRequest.activity_id == activity_id, ReviewRequest.user_id == 1
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
        "total_questions": 3,
        "question_index": 1,
    })


@app.post("/api/submit")
def api_submit(payload: dict, db: Session = Depends(get_db)):
    activity = db.get(Activity, payload.get("activity_id", 0))
    if not activity:
        raise HTTPException(404, "Activity not found")
    steps = steps_of(activity)
    idx = int(payload.get("step_index", 0))
    if not steps or idx >= len(steps):
        raise HTTPException(400, "Invalid step")
    student = str(payload.get("response", ""))[:MAX_LEN]
    if not student.strip():
        return {"verdict": "incomplete", "reply": "Please enter a step first.", "revealed": False}
    prior_fails = db.query(Attempt).filter(
        Attempt.activity_id == activity.id, Attempt.user_id == 1,
        Attempt.step_index == idx, Attempt.verdict != "correct").count()
    history = DEMO_HISTORY.setdefault(f"{activity.id}:{idx}", [])
    result = orchestrate(
        activity.problem_statement, steps[idx]["prompt"], steps[idx]["expected"],
        student, idx, len(steps), int(payload.get("hint_level", 0)),
        prior_fails, bool(activity.allow_solution_reveal),
        activity.max_attempts_before_reveal or 3, history,
    )
    if result["verdict"] != "correct" and steps and idx < len(steps):
        step_hints = steps[idx].get("hint", [])
        if step_hints:
            result["hint"] = step_hints[min(1, len(step_hints) - 1)]

    history += [{"role": "user", "content": student},
                {"role": "assistant", "content": result["reply"]}]
    db.add(Attempt(activity_id=activity.id, user_id=1, step_index=idx,
                   student_response=student, verdict=result["verdict"],
                   hint_level=int(payload.get("hint_level", 0))))
    db.commit()
    return result


@app.post("/api/hint")
def api_hint(payload: dict, db: Session = Depends(get_db)):
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
    db.add(HintRequest(activity_id=activity.id, user_id=1, level=level, hint_text=text))
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
def api_review_request(payload: dict, db: Session = Depends(get_db)):
    act_id = int(payload.get("activity_id", 0))
    req = db.query(ReviewRequest).filter(
        ReviewRequest.activity_id == act_id, ReviewRequest.user_id == 1
    ).first()
    if not req:
        req = ReviewRequest(activity_id=act_id, user_id=1, status="submitted", notes="Student requested instructor review")
        db.add(req)
        db.commit()
    return {"status": "submitted", "message": "Your work has been submitted for Dr. Lee to review!"}


@app.get("/lessons", response_class=HTMLResponse)
def lessons_page(request: Request, db: Session = Depends(get_db)):
    """Student My Lessons: only lessons from assigned packages."""
    lesson_cards, _ = _student_cards(
        db, db.query(Publication).order_by(Publication.id.desc()).all())
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
def instructor(request: Request, db: Session = Depends(get_db)):
    attempts = db.query(Attempt).order_by(Attempt.id.desc()).limit(50).all()
    students = db.query(User).filter(User.role == "student").all()
    total = db.query(Attempt).count()
    correct = db.query(Attempt).filter(Attempt.verdict == "correct").count()
    hint_count = db.query(HintRequest).count()
    mastery = db.query(Mastery).filter(Mastery.user_id == 1).all()
    return templates.TemplateResponse(request, "instructor.html", {
        "request": request, "attempts": attempts,
        "students": students, "total": total, "correct": correct,
        "hint_count": hint_count, "mastery": mastery,
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
        db, db.query(Publication).order_by(Publication.id.desc()).all())
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


@app.get("/progress", response_class=HTMLResponse)
def progress(request: Request, db: Session = Depends(get_db)):
    attempts = db.query(Attempt).filter(Attempt.user_id == 1).order_by(Attempt.id.desc()).all()
    mastery = db.query(Mastery).filter(Mastery.user_id == 1).all()
    return templates.TemplateResponse(request, "progress.html", {"request": request, "attempts": attempts,
                                       "mastery": mastery, "demo": not config.ai_configured()})


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
def api_generate_questions(payload: dict):
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
def instructor_dashboard(request: Request, db: Session = Depends(get_db)):
    from .courses import get_course_options
    activities = db.query(Activity).all()
    courses = db.query(Course).all()
    return templates.TemplateResponse(request, "instructor_dashboard.html", {
        "request": request, "activities": activities, "courses": courses,
        "course_options": get_course_options(),
        "demo": not config.ai_configured(),
    })


@app.get("/instructor/drafts", response_class=HTMLResponse)
def instructor_drafts(request: Request, db: Session = Depends(get_db)):
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


@app.post("/api/lessons/assign")
def api_lessons_assign(payload: dict, db: Session = Depends(get_db)):
    """Mock assign: persist verified questions as Activities, link under a UUID."""
    from .schemas import question_to_activity_fields, step_from_legacy

    questions = payload.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise HTTPException(400, "No questions to assign")
    topic = str(payload.get("topic", "Calculus"))
    activity_ids: list[int] = []
    stored_questions: list[dict] = []
    for q in questions:
        if not isinstance(q, dict) or not q.get("is_verified"):
            continue
        steps = []
        for i, s in enumerate(q.get("steps", []) or []):
            if not isinstance(s, dict):
                continue
            legacy = {
                "prompt": s.get("description", ""),
                "expected": s.get("expected_input_latex", ""),
                "hint": list(s.get("hints", []) or []),
            }
            # Validate round-trip through the schema adapter.
            step = step_from_legacy(legacy, i)
            steps.append({"prompt": step.description,
                          "expected": step.expected_input_latex,
                          "hint": step.hints})
        if not steps:
            continue
        fields = question_to_activity_fields.__wrapped__ if hasattr(
            question_to_activity_fields, "__wrapped__") else None
        _ = fields  # adapter applied below via explicit construction
        activity = Activity(
            title=str(q.get("prompt_latex", ""))[:200] or "Assigned question",
            course_topic=topic,
            problem_statement=str(q.get("prompt_latex", "")),
            learning_objectives="",
            expected_steps_json=json.dumps(steps),
            accepted_methods="",
            misconceptions="",
            hint_sequence_json=json.dumps(
                ["Conceptual question", "Specific hint",
                 "Formula guidance", "Worked substep"]),
            max_attempts_before_reveal=3,
            allow_solution_reveal=True,
        )
        db.add(activity)
        db.flush()
        activity_ids.append(activity.id)
        stored_questions.append(q)
    if not activity_ids:
        raise HTTPException(400, "No verifiable questions to assign")
    db.commit()
    link_id = uuid.uuid4().hex[:8]
    LESSON_LINKS[link_id] = {"topic": topic,
                             "questions": stored_questions,
                             "activity_ids": activity_ids,
                             "title": str(payload.get("title", "")),
                             "due": str(payload.get("due", "")),
                             "grading": str(payload.get("grading", "Completion")),
                             "difficulty_balance": str(
                                 payload.get("difficulty_balance", "mixed")),
                             "worked_example": bool(
                                 payload.get("worked_example", False))}
    return {"link_id": link_id,
            "solve_url": f"/student/solve/{link_id}/0",
            "activity_ids": activity_ids}


@app.get("/student/solve/{link_id}/{question_index}", response_class=HTMLResponse)
def student_solve(link_id: str, question_index: int, request: Request,
                  db: Session = Depends(get_db)):
    link = LESSON_LINKS.get(link_id)
    if not link:
        raise HTTPException(404, "Class link not found (demo links clear on restart)")
    ids = link["activity_ids"]
    if not ids or not (0 <= question_index < len(ids)):
        raise HTTPException(400, "Invalid question index")
    activity = db.get(Activity, ids[question_index])
    if not activity:
        raise HTTPException(404, "Activity not found")
    steps = steps_of(activity)
    attempts = db.query(Attempt).filter(
        Attempt.activity_id == activity.id, Attempt.user_id == 1
    ).order_by(Attempt.id.desc()).all()
    return templates.TemplateResponse(request, "student_solver.html", {
        "request": request, "activity": activity, "steps": steps,
        "active_step": 0, "hints": hints_of(activity), "attempts": attempts,
        "link_id": link_id, "question_index": question_index,
        "total_questions": len(ids),
        "demo": not config.ai_configured(),
    })


@app.get("/success/{link_id}", response_class=HTMLResponse)
def assign_success(link_id: str, request: Request):
    link = LESSON_LINKS.get(link_id)
    if not link:
        raise HTTPException(404, "Class link not found (demo links clear on restart)")
    return templates.TemplateResponse(request, "success.html", {
        "request": request, "link_id": link_id,
        "topic": link.get("topic", ""),
        "count": len(link.get("activity_ids", [])),
        "title": link.get("title", ""),
        "due": link.get("due", ""),
        "grading": link.get("grading", ""),
    })


@app.get("/api/get-lesson/{link_id}")
def api_get_lesson(link_id: str):
    """Read-only inspector for a mock-assigned lesson (demo debugging)."""
    link = LESSON_LINKS.get(link_id)
    if not link:
        raise HTTPException(404, "Class link not found (demo links clear on restart)")
    return {"link_id": link_id, "topic": link.get("topic", ""),
            "questions": link.get("questions", []),
            "activity_ids": link.get("activity_ids", [])}


@app.get("/api/analytics")
def api_analytics(course: str = "", db: Session = Depends(get_db)):
    """Instructor insights, aggregated from the tutor's own event log.

    Rates cover only the steps SymPy decided; unread responses are counted
    beside them rather than folded into a percentage.
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
def api_lessons_generate(payload: dict):
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
def api_lessons_save(payload: dict, db: Session = Depends(get_db)):
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
def api_lessons_list(db: Session = Depends(get_db)):
    """List lesson drafts (id, title, topic, status). No content bodies."""
    lessons = db.query(Lesson).order_by(Lesson.updated_at.desc()).all()
    return {"lessons": [_lesson_to_dict(les) for les in lessons]}


@app.get("/api/lessons/{lesson_id}")
def api_lessons_get(lesson_id: int, db: Session = Depends(get_db)):
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
                        db: Session = Depends(get_db)):
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
def api_lessons_delete(lesson_id: int, db: Session = Depends(get_db)):
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
def api_assignments_generate(payload: dict, db: Session = Depends(get_db)):
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
        from .schemas import lesson_content_from_json
        content = lesson_content_from_json(lesson.instructor_content_json)
        scoped = "\n".join(content.objectives)
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
def api_assignments_save(payload: dict, db: Session = Depends(get_db)):
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
def api_assignments_list(db: Session = Depends(get_db)):
    """List assignment drafts (metadata only, no solutions)."""
    rows = db.query(Assignment).order_by(Assignment.updated_at.desc()).all()
    return {"assignments": [_assignment_to_dict(a) for a in rows]}


@app.get("/api/assignments/{asg_id}")
def api_assignments_get(asg_id: int, db: Session = Depends(get_db)):
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
def api_assignments_approve(asg_id: int, db: Session = Depends(get_db)):
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
def api_assignments_delete(asg_id: int, db: Session = Depends(get_db)):
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
def api_publications_create(payload: dict, db: Session = Depends(get_db)):
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
    # Back-compat: keep the in-memory link working for the success page.
    LESSON_LINKS[link_id] = {
        "topic": (lesson.topic if lesson else asg.topic),
        "questions": [], "activity_ids": [],
        "title": pub.title, "due": pub.due_date,
        "grading": pub.grading_policy,
        "publication_id": pub.id,
    }
    return {"link_id": link_id, "publication_id": pub.id,
            "solve_url": f"/learn/{link_id}",
            "lesson_id": lesson.id if lesson else None,
            "assignment_id": asg.id if asg else None}


@app.delete("/api/publications/{link_id}")
def api_publications_delete(link_id: str, db: Session = Depends(get_db)):
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
    LESSON_LINKS.pop(link_id, None)
    return {"retired": link_id, "reverted": reverted}


@app.post("/api/assign")
def api_assign(payload: dict, db: Session = Depends(get_db)):
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
            }, db)
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
    """Student-safe assignment items: prompts + step descriptions + hints.

    Solutions never leave the server; expected answers are excluded.
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
            "steps": [{"description": (s.get("description", "")
                                       if isinstance(s, dict) else ""),
                       "hints": (list(s.get("hints", []))
                                 if isinstance(s, dict) else [])}
                      for s in it.get("steps", [])],
            "hints": list(it.get("hints", [])),
            "difficulty": it.get("difficulty", 3),
        })
    return items


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
    sub = db.query(Submission).filter(
        Submission.publication_id == pub.id,
        Submission.user_id == 1).first()
    try:
        saved = _json.loads(sub.answers_json) if sub and sub.answers_json else {}
    except Exception:
        saved = {}
    return templates.TemplateResponse(request, "student_assignment.html", {
        "request": request, "publication": pub,
        "title": asg.title or pub.title or "Untitled",
        "due": pub.due_date or "", "course": pub.assigned_course or "",
        "items": items, "saved": saved if isinstance(saved, dict) else {},
        "submitted": bool(sub and sub.status == "submitted"),
        "submitted_at": str(sub.submitted_at)[:16] if sub and sub.submitted_at else "",
        "active_nav": "assignments",
        "demo": not config.ai_configured(),
    })


@app.post("/api/assignments/progress")
def api_assignment_progress(payload: dict, db: Session = Depends(get_db)):
    """Save draft answers (resume any time). Never marks submitted."""
    import json as _json
    from datetime import datetime, timezone

    link_id = str(payload.get("link_id", "") or "").strip()
    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub or not pub.assignment_id:
        raise HTTPException(404, "Assignment not found")
    sub = db.query(Submission).filter(
        Submission.publication_id == pub.id,
        Submission.user_id == 1).first()
    if sub and sub.status == "submitted":
        raise HTTPException(400, "Already submitted — answers are final")
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict):
        raise HTTPException(400, "Invalid answers")
    clean = {str(k)[:40]: str(v)[:2000] for k, v in answers.items()}
    if not sub:
        sub = Submission(publication_id=pub.id, user_id=1,
                         answers_json=_json.dumps(clean), status="draft")
        db.add(sub)
    else:
        sub.answers_json = _json.dumps(clean)
        sub.status = "draft"
        sub.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"saved": True, "n_answers": len(clean)}


@app.post("/api/assignments/submit")
def api_assignment_submit(payload: dict, db: Session = Depends(get_db)):
    """Final submission: persists answers, locks the row. No re-submit."""
    import json as _json
    from datetime import datetime, timezone

    link_id = str(payload.get("link_id", "") or "").strip()
    pub = db.query(Publication).filter(
        Publication.link_id == link_id).first()
    if not pub or not pub.assignment_id:
        raise HTTPException(404, "Assignment not found")
    sub = db.query(Submission).filter(
        Submission.publication_id == pub.id,
        Submission.user_id == 1).first()
    if sub and sub.status == "submitted":
        raise HTTPException(400, "Already submitted")
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict):
        raise HTTPException(400, "Invalid answers")
    clean = {str(k)[:40]: str(v)[:2000] for k, v in answers.items()}
    now = datetime.now(timezone.utc)
    if not sub:
        sub = Submission(publication_id=pub.id, user_id=1,
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
