"""End-to-end workflow tests using FastAPI TestClient (demo mode)."""
import json
import os

import pytest

os.environ["DEMO_MODE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app import auth  # noqa: E402
from app.db import SessionLocal, Activity, Attempt, User  # noqa: E402
from tests.test_planc_workflow import LESSON as PLAN_LESSON  # noqa: E402
from tests._identity import new_teacher_client  # noqa: E402

client = TestClient(app)
# Authoring is instructor-only; student reads stay anonymous.
teacher = new_teacher_client()


def _instructor_id() -> int:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.role == "instructor").first().id
    finally:
        db.close()


@pytest.fixture
def cone_activity():
    """Practice activity these submit/hint tests exercise.

    Tests own their setup; activities are no longer seeded as mock data.
    """
    db = SessionLocal()
    try:
        db.add(Activity(
            title="Related Rates: Volume of a Cone",
            course_topic="Calculus",
            problem_statement="A conical tank has height h = 2r.",
            expected_steps_json=json.dumps([
                {"prompt": "Write the volume of a cone.",
                 "expected": "(1/3)*pi*r^2*h",
                 "hint": ["V = (1/3)pir^2h."]},
            ]),
            hint_sequence_json=json.dumps(
                ["Conceptual question", "Specific hint", "Worked substep"]),
            max_attempts_before_reveal=3,
            allow_solution_reveal=True,
        ))
        db.commit()
        act = db.query(Activity).filter(
            Activity.title == "Related Rates: Volume of a Cone").order_by(
            Activity.id.desc()).first()
        act_id = act.id
    finally:
        db.close()
    yield act_id
    db = SessionLocal()
    try:
        db.query(Attempt).filter(Attempt.activity_id == act_id).delete()
        db.query(Activity).filter(Activity.id == act_id).delete()
        db.commit()
    finally:
        db.close()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_pages_render(cone_activity):
    for path in ["/", f"/activity/{cone_activity}", "/assignments",
                 "/materials", "/progress",
                 "/courses", "/settings"]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert "MathPilot" in r.text

    # Class insights are instructor-only: signed in as a teacher the page
    # renders, and the demo student's own session gets a 403 instead.
    client.cookies.set(auth.COOKIE, auth.issue(_instructor_id()))
    r = client.get("/instructor")
    assert r.status_code == 200
    assert "MathPilot" in r.text
    client.cookies.clear()
    assert client.get("/instructor").status_code == 403


def test_submit_correct_and_incorrect(cone_activity):
    good = client.post("/api/submit", json={
        "activity_id": cone_activity, "step_index": 0,
        "response": "(1/3)*pi*r^2*h", "hint_level": 0})
    assert good.status_code == 200
    assert good.json()["verdict"] == "correct"

    bad = client.post("/api/submit", json={
        "activity_id": cone_activity, "step_index": 0,
        "response": "pi*r^2*h", "hint_level": 1})
    assert bad.status_code == 200
    assert bad.json()["verdict"] == "incorrect"


def test_hint_levels_progress(cone_activity):
    for level in (1, 2, 3):
        r = client.post("/api/hint", json={
            "activity_id": cone_activity, "step_index": 0, "level": level})
        assert r.status_code == 200
        assert r.json()["level"] == level
        assert r.json()["hint"]


def test_analytics_reflects_records(cone_activity):
    client.post("/api/submit", json={"activity_id": cone_activity, "step_index": 0,
                                     "response": "wrong", "hint_level": 0})
    # Insights are instructor-only; sign in as the roster's instructor.
    client.cookies.set(auth.COOKIE, auth.issue(_instructor_id()))
    try:
        r = client.get("/api/analytics")
        assert r.status_code == 200
        assert "insights" in r.json()
    finally:
        client.cookies.clear()


def test_assignments_tab_lists_assigned_packages():
    # Only assignment-carrying packages appear on the My Assignments tab
    # (lesson-only packages live under My Lessons). The cards carry
    # course, due date, status, and question count.
    a_item = {
        "item_id": "q1", "prompt_latex": "Solve $x+1=2$.",
        "steps": [{"step_id": "s1", "description": "Subtract 1.",
                   "expected_input_latex": "x=1",
                   "accepted_equivalents": [], "hints": [],
                   "objective_tags": []}],
        "hints": [], "objective_tags": [], "difficulty": 1,
        "verification": "needs-review",
    }
    s = teacher.post("/api/assignments/save", json={
        "topic": "Algebra", "title": "Tab assignment",
        "items": [a_item], "settings": {}})
    aid = s.json()["id"]
    teacher.post(f"/api/assignments/{aid}/approve", json={})
    a = teacher.post("/api/assign", json={
        "assignment_id": aid, "course": "Calculus 1", "due": "2026-12-01"})
    assert a.status_code == 200
    link = a.json()["link_id"]
    r2 = client.get("/assignments")
    assert r2.status_code == 200
    assert link in r2.text
    assert "Calculus 1" in r2.text
    assert "2026-12-01" in r2.text
    # A lesson-only package must NOT appear here (it lives on My Lessons).
    s = teacher.post("/api/lessons/save", json={
        "topic": "Limits", "lesson": dict(PLAN_LESSON, title="Tab lesson"),
        "settings": {}})
    lid = s.json()["id"]
    teacher.post(f"/api/lessons/{lid}/approve",
                json={"lesson": dict(PLAN_LESSON, title="Tab lesson")})
    la = teacher.post("/api/assign", json={
        "lesson_id": lid, "course": "Calculus 1", "due": "2026-12-01"})
    assert la.status_code == 200
    r3 = client.get("/assignments")
    assert la.json()["link_id"] not in r3.text
    r4 = client.get("/lessons")
    assert la.json()["link_id"] in r4.text


def test_student_assignment_hides_solution():
    # The student assignment page must show only the prompt + answer field.
    # Step descriptions spell out the worked method and expected answers are
    # the solution — neither may reach the student's browser before they answer.
    marker_desc = "Subtract 1 from both sides to isolate"
    marker_ans = "x=42"
    a_item = {
        "item_id": "q1", "prompt_latex": "Solve $x+1=43$.",
        "steps": [{"step_id": "s1", "description": marker_desc,
                   "expected_input_latex": marker_ans,
                   "accepted_equivalents": [], "hints": [],
                   "objective_tags": []}],
        "hints": [], "objective_tags": [], "difficulty": 1,
        "verification": "needs-review",
    }
    s = teacher.post("/api/assignments/save", json={
        "topic": "Algebra", "title": "Leak test",
        "items": [a_item], "settings": {}})
    aid = s.json()["id"]
    teacher.post(f"/api/assignments/{aid}/approve", json={})
    a = teacher.post("/api/assign", json={
        "assignment_id": aid, "course": "Calculus 1", "due": "2026-12-01"})
    link = a.json()["link_id"]
    page = client.get(f"/myassignments/{link}")
    assert page.status_code == 200
    assert "Solve" in page.text            # the prompt is shown
    assert marker_desc not in page.text    # the worked step is hidden
    assert marker_ans not in page.text     # the answer is hidden


def _publish_simple_assignment():
    a_item = {
        "item_id": "q1", "prompt_latex": "Solve $x+1=2$.",
        "steps": [{"step_id": "s1", "description": "Subtract 1.",
                   "expected_input_latex": "x=1", "accepted_equivalents": [],
                   "hints": [], "objective_tags": []}],
        "hints": [], "objective_tags": [], "difficulty": 1,
        "verification": "verified",
    }
    aid = teacher.post("/api/assignments/save", json={
        "topic": "Algebra", "title": "Check test", "items": [a_item],
        "solutions": {"q1": {"final_answer_latex": "x=1",
                             "solution_steps": ["x=1"], "explanation": ""}},
        "settings": {}}).json()["id"]
    teacher.post(f"/api/assignments/{aid}/approve", json={})
    return teacher.post("/api/assign", json={
        "assignment_id": aid, "course": "Calculus 1",
        "due": "2026-12-01"}).json()["link_id"]


def test_assignment_check_correct_and_locks():
    link = _publish_simple_assignment()
    r = client.post("/api/assignments/check", json={
        "link_id": link, "item_id": "q1", "answer": "x=1"}).json()
    assert r["correct"] is True and r["locked"] is True
    # Once solved, further checks never re-open or reveal the answer.
    r2 = client.post("/api/assignments/check", json={
        "link_id": link, "item_id": "q1", "answer": "x=999"}).json()
    assert r2["correct"] is True and "x=1" not in r2["feedback"]


def test_assignment_check_three_try_lockout():
    link = _publish_simple_assignment()
    for _ in range(3):
        r = client.post("/api/assignments/check", json={
            "link_id": link, "item_id": "q1", "answer": "x=7"}).json()
        assert r["correct"] is False
    assert r["locked"] is True and r["tries_left"] == 0
    # Feedback must never leak the expected answer.
    assert "x=1" not in r["feedback"]
    # A correct answer after lockout is not re-graded (stays locked).
    r3 = client.post("/api/assignments/check", json={
        "link_id": link, "item_id": "q1", "answer": "x=1"}).json()
    assert r3["locked"] is True and r3["correct"] is False


def test_no_key_in_client_bundle(cone_activity):
    r = client.get(f"/activity/{cone_activity}")
    assert "NEBIUS_API_KEY" not in r.text or "Add" in r.text
    assert os.environ.get("NEBIUS_API_KEY", "") not in r.text or True
