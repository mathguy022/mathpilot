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
from .db import get_db, init_db, Activity, Attempt, Course, CourseMaterial, HintRequest, Mastery, User
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
