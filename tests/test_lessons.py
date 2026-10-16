"""Tests for mock lesson assign + class-link solver pages."""
import os

os.environ["DEMO_MODE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

QUESTION = {
    "prompt_latex": "Solve $x+1=2$.",
    "final_answer_latex": "x=1",
    "difficulty": 1,
    "source": "ai-generated",
    "is_verified": True,
    "steps": [{
        "step_id": "gen-1",
        "description": "Subtract 1.",
        "expected_input_latex": "x=1",
        "accepted_equivalents": [],
        "hints": ["What undoes +1?"],
        "objective_tags": [],
    }],
}


def test_dashboard_renders():
    r = client.get("/instructor/dashboard")
    assert r.status_code == 200
    assert "Generate Lesson" in r.text
    assert "Create Assignment" in r.text
    assert "Workspace Canvas" in r.text
    assert "Review & Publish" in r.text


def test_success_and_inspector():
    r = client.post("/api/lessons/assign",
                    json={"topic": "Algebra", "questions": [QUESTION],
                          "title": "Week 3", "due": "2026-10-22",
                          "grading": "Mastery",
                          "difficulty_balance": "mixed",
                          "worked_example": True})
    link_id = r.json()["link_id"]
    page = client.get(f"/success/{link_id}")
    assert page.status_code == 200
    assert link_id in page.text
    assert "Week 3" in page.text
    insp = client.get(f"/api/get-lesson/{link_id}")
    assert insp.status_code == 200
    assert insp.json()["topic"] == "Algebra"
    assert client.get("/api/get-lesson/deadbeef").status_code == 404
    assert client.get("/success/deadbeef").status_code == 404


def test_assign_rejects_unverified():
    bad = dict(QUESTION, is_verified=False)
    r = client.post("/api/lessons/assign",
                    json={"topic": "Algebra", "questions": [bad]})
    assert r.status_code == 400


def test_assign_and_solve_roundtrip():
    r = client.post("/api/lessons/assign",
                    json={"topic": "Algebra", "questions": [QUESTION]})
    assert r.status_code == 200
    body = r.json()
    assert "link_id" in body and "solve_url" in body
    assert len(body["activity_ids"]) == 1

    activity_id = body["activity_ids"][0]
    page = client.get(f"/student/solve/{body['link_id']}/0")
    assert page.status_code == 200
    assert "student-input" in page.text
    assert "Step 1 of 1" in page.text

    submit = client.post("/api/submit", json={
        "activity_id": activity_id, "step_index": 0,
        "response": "x=1", "hint_level": 0})
    assert submit.status_code == 200
    assert submit.json()["verdict"] == "correct"


def test_bad_link_404():
    r = client.get("/student/solve/deadbeef/0")
    assert r.status_code == 404
