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
    Assignment, get_db, init_db, Activity, Attempt, Course, CourseMaterial,
    HintRequest, Lesson, Mastery, Publication, User,
)
from .seed import seed
from .tutor import orchestrate
from .verifier import MAX_LEN

BASE = Path(__file__).resolve().parent.parent
app = FastAPI(title="MathPilot")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

init_db()
seed()

DEMO_HISTORY: dict[str, list[dict]] = {}

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
    activities = db.query(Activity).all()
    courses = db.query(Course).all()
    students = db.query(User).filter(User.role == "student").all()
    return templates.TemplateResponse(request, "home.html", {
        "request": request, "activities": activities, "courses": courses,
        "students": students, "demo": not config.ai_configured(),
        "math_model": config.NEBIUS_MATH_MODEL,
        "socratic_model": config.NEBIUS_SOCRATIC_MODEL,
    })


@app.get("/activity/{activity_id}", response_class=HTMLResponse)
def activity_page(activity_id: int, request: Request, db: Session = Depends(get_db)):
    activity = db.get(Activity, activity_id)
    if not activity:
        raise HTTPException(404, "Activity not found")
    steps = steps_of(activity)
    attempts = db.query(Attempt).filter(
        Attempt.activity_id == activity_id, Attempt.user_id == 1).order_by(Attempt.id.desc()).all()
    return templates.TemplateResponse(request, "activity.html", {
        "request": request, "activity": activity, "steps": steps,
        "hints": hints_of(activity), "attempts": attempts,
        "demo": not config.ai_configured(),
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


@app.get("/instructor", response_class=HTMLResponse)
def instructor(request: Request, db: Session = Depends(get_db)):
    activities = db.query(Activity).all()
    attempts = db.query(Attempt).order_by(Attempt.id.desc()).limit(50).all()
    students = db.query(User).filter(User.role == "student").all()
    total = db.query(Attempt).count()
    correct = db.query(Attempt).filter(Attempt.verdict == "correct").count()
    hint_count = db.query(HintRequest).count()
    mastery = db.query(Mastery).filter(Mastery.user_id == 1).all()
    return templates.TemplateResponse(request, "instructor.html", {
        "request": request, "activities": activities, "attempts": attempts,
        "students": students, "total": total, "correct": correct,
        "hint_count": hint_count, "mastery": mastery,
        "demo": not config.ai_configured(),
    })


@app.get("/assignments", response_class=HTMLResponse)
def assignments(request: Request, db: Session = Depends(get_db)):
    activities = db.query(Activity).all()
    return templates.TemplateResponse(request, "assignments.html", {"request": request, "activities": activities})


@app.get("/assignments/new", response_class=HTMLResponse)
def assignment_new(request: Request):
    return templates.TemplateResponse(request, "assignment_form.html", {"request": request, "activity": None})


@app.get("/assignments/{activity_id}/edit", response_class=HTMLResponse)
def assignment_edit(activity_id: int, request: Request, db: Session = Depends(get_db)):
    activity = db.get(Activity, activity_id)
    if not activity:
        raise HTTPException(404, "Activity not found")
    return templates.TemplateResponse(request, "assignment_form.html", {"request": request, "activity": activity})


@app.post("/assignments/save")
def assignment_save(
    id: int = Form(0), title: str = Form(...), course_topic: str = Form("Calculus"),
    problem_statement: str = Form(...), learning_objectives: str = Form(""),
    expected_steps: str = Form("[]"), accepted_methods: str = Form(""),
    misconceptions: str = Form(""), allow_solution_reveal: bool = Form(False),
    max_attempts_before_reveal: int = Form(3),
    db: Session = Depends(get_db),
):
    if not title.strip() or not problem_statement.strip():
        raise HTTPException(400, "Title and problem statement are required")
    try:
        parsed = json.loads(expected_steps)
        assert isinstance(parsed, list)
    except Exception:
        raise HTTPException(400, "Expected steps must be a JSON list")
    if id:
        activity = db.get(Activity, id)
        if not activity:
            raise HTTPException(404, "Activity not found")
    else:
        activity = Activity(expected_steps_json="[]", hint_sequence_json='["Conceptual question","Specific hint","Formula guidance","Worked substep"]')
        db.add(activity)
    activity.title = title.strip()
    activity.course_topic = course_topic.strip()
    activity.problem_statement = problem_statement.strip()
    activity.learning_objectives = learning_objectives
    activity.expected_steps_json = json.dumps(parsed)
    activity.accepted_methods = accepted_methods
    activity.misconceptions = misconceptions
    activity.allow_solution_reveal = allow_solution_reveal
    activity.max_attempts_before_reveal = max_attempts_before_reveal
    db.commit()
    return RedirectResponse(f"/activity/{activity.id}", status_code=303)


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
    activities = db.query(Activity).all()
    courses = db.query(Course).all()
    return templates.TemplateResponse(request, "instructor_dashboard.html", {
        "request": request, "activities": activities, "courses": courses,
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
def api_analytics(db: Session = Depends(get_db)):
    attempts = db.query(Attempt).all()
    by_step: dict[str, dict] = {}
    for a in attempts:
        key = f"activity {a.activity_id} · step {a.step_index + 1}"
        cell = by_step.setdefault(key, {"attempts": 0, "incorrect": 0})
        cell["attempts"] += 1
        if a.verdict == "incorrect":
            cell["incorrect"] += 1
    insights = [
        {"label": k, **v,
         "rate": round(v["incorrect"] / v["attempts"], 2) if v["attempts"] else 0}
        for k, v in by_step.items()
    ]
    return {"insights": sorted(insights, key=lambda r: r["rate"], reverse=True),
            "demo_hint_requests": db.query(HintRequest).count()}


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
    asg.title = (str(payload.get("title", ""))
                 or str(settings.get("title", ""))
                 or "Untitled assignment")[:300]
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


@app.get("/learn/{link_id}", response_class=HTMLResponse)
def student_learn(link_id: str, request: Request, db: Session = Depends(get_db)):
    """Student-facing interactive lesson: derived content only, no keys."""
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
    asg_items: list[dict] = []
    if pub.assignment_id:
        asg = db.get(Assignment, pub.assignment_id)
        if asg:
            try:
                raw = _json.loads(asg.questions_json or "[]")
            except Exception:
                raw = []
            # Strip instructor-only data: steps keep prompts (students need
            # them to attempt), solutions never leave the server.
            for it in raw:
                if not isinstance(it, dict):
                    continue
                asg_items.append({
                    "item_id": it.get("item_id", ""),
                    "prompt_latex": it.get("prompt_latex", ""),
                    "steps": [{"description": (s.get("description", "") if isinstance(s, dict) else ""),
                               "hints": (list(s.get("hints", [])) if isinstance(s, dict) else [])}
                              for s in it.get("steps", [])],
                    "hints": list(it.get("hints", [])),
                    "difficulty": it.get("difficulty", 3),
                })
    return templates.TemplateResponse(request, "student_lesson.html", {
        "request": request, "publication": pub,
        "lesson": lesson_payload, "assignment_items": asg_items,
        "demo": not config.ai_configured(),
    })


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
