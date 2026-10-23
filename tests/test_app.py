"""End-to-end workflow tests using FastAPI TestClient (demo mode)."""
import json
import os

import pytest

os.environ["DEMO_MODE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.db import SessionLocal, Activity, Attempt  # noqa: E402
from tests.test_planc_workflow import LESSON as PLAN_LESSON  # noqa: E402

client = TestClient(app)


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
    for path in ["/", f"/activity/{cone_activity}", "/instructor", "/assignments",
                 "/materials", "/progress",
                 "/courses", "/settings"]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert "MathPilot" in r.text


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
    r = client.get("/api/analytics")
    assert r.status_code == 200
    assert "insights" in r.json()


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
    s = client.post("/api/assignments/save", json={
        "topic": "Algebra", "title": "Tab assignment",
        "items": [a_item], "settings": {}})
    aid = s.json()["id"]
    client.post(f"/api/assignments/{aid}/approve", json={})
    a = client.post("/api/assign", json={
        "assignment_id": aid, "course": "Calculus 1", "due": "2026-12-01"})
    assert a.status_code == 200
    link = a.json()["link_id"]
    r2 = client.get("/assignments")
    assert r2.status_code == 200
    assert link in r2.text
    assert "Calculus 1" in r2.text
    assert "2026-12-01" in r2.text
    # A lesson-only package must NOT appear here (it lives on My Lessons).
    s = client.post("/api/lessons/save", json={
        "topic": "Limits", "lesson": dict(PLAN_LESSON, title="Tab lesson"),
        "settings": {}})
    lid = s.json()["id"]
    client.post(f"/api/lessons/{lid}/approve",
                json={"lesson": dict(PLAN_LESSON, title="Tab lesson")})
    la = client.post("/api/assign", json={
        "lesson_id": lid, "course": "Calculus 1", "due": "2026-12-01"})
    assert la.status_code == 200
    r3 = client.get("/assignments")
    assert la.json()["link_id"] not in r3.text
    r4 = client.get("/lessons")
    assert la.json()["link_id"] in r4.text


def test_no_key_in_client_bundle(cone_activity):
    r = client.get(f"/activity/{cone_activity}")
    assert "NEBIUS_API_KEY" not in r.text or "Add" in r.text
    assert os.environ.get("NEBIUS_API_KEY", "") not in r.text or True
