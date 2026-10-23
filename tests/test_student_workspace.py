"""Tests for the Redesigned Student Learning Workspace and supporting routes."""
import json
import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal, Activity, Attempt, ReviewRequest

client = TestClient(app)


@pytest.fixture
def chain_activity():
    """Create the chain-rule practice activity these tests exercise.

    Activities are instructor-created content, so tests own their setup
    rather than relying on seeded mock data.
    """
    db = SessionLocal()
    try:
        db.add(Activity(
            title="Chain rule practice",
            course_topic="Calculus I",
            problem_statement="Differentiate the function:\n$$f(x) = (3x + 1)^2$$",
            learning_objectives="Identify inner function, apply chain rule, and simplify.",
            expected_steps_json=json.dumps([
                {"prompt": "Identify inner function u = g(x).",
                 "expected": "u = 3x + 1",
                 "hint": ["Look at what is inside the parentheses."]},
                {"prompt": "Apply the chain rule.",
                 "expected": "2(3x + 1)(3)",
                 "hint": ["Multiply by the derivative of the inner function."]},
                {"prompt": "Simplify.",
                 "expected": "18x + 6",
                 "hint": ["Distribute."]},
            ]),
            hint_sequence_json=json.dumps(
                ["Conceptual question", "Specific hint", "Worked substep"]),
            max_attempts_before_reveal=3,
            allow_solution_reveal=True,
        ))
        db.commit()
        act = db.query(Activity).filter(
            Activity.title == "Chain rule practice").order_by(
            Activity.id.desc()).first()
        act_id = act.id
    finally:
        db.close()
    yield act_id
    db = SessionLocal()
    try:
        db.query(Attempt).filter(Attempt.activity_id == act_id).delete()
        db.query(ReviewRequest).filter(
            ReviewRequest.activity_id == act_id).delete()
        db.query(Activity).filter(Activity.id == act_id).delete()
        db.commit()
    finally:
        db.close()


def test_practice_page_renders():
    r = client.get("/practice", follow_redirects=True)
    assert r.status_code == 200
    assert "Practice with AI Tutor" in r.text


def test_student_routes_render():
    routes = [
        "/practice",
        "/dashboard",
        "/lessons",
        "/tutor",
        "/study-guides",
        "/help",
        "/courses",
        "/assignments",
        "/progress",
    ]
    for route in routes:
        r = client.get(route, follow_redirects=True)
        assert r.status_code == 200, f"Route {route} failed with {r.status_code}"
        assert "MathPilot" in r.text


def test_chain_rule_workspace_render(chain_activity):
    r = client.get(f"/activity/{chain_activity}")
    assert r.status_code == 200
    assert "Chain rule practice" in r.text
    assert "Approved by Dr. Lee" in r.text
    assert "AI Math Tutor" in r.text
    assert "Course notes" in r.text
    assert "Instructor review" in r.text
    assert "Math keyboard" in r.text
    assert "Your working" in r.text


def test_chain_rule_step_verification(chain_activity):
    act_id = chain_activity

    # Step 0: Identify inner function u = 3x + 1
    r0 = client.post("/api/submit", json={
        "activity_id": act_id,
        "step_index": 0,
        "response": "u = 3x + 1",
        "hint_level": 0
    })
    assert r0.status_code == 200
    assert r0.json()["verdict"] == "correct"

    # Step 1: Missing derivative of inner function -> incorrect
    r1_bad = client.post("/api/submit", json={
        "activity_id": act_id,
        "step_index": 1,
        "response": "f'(x) = 2(3x + 1)",
        "hint_level": 1
    })
    assert r1_bad.status_code == 200
    assert r1_bad.json()["verdict"] == "incorrect"
    assert "hint" in r1_bad.json()

    # Step 1: Correct chain rule application -> correct
    r1_good = client.post("/api/submit", json={
        "activity_id": act_id,
        "step_index": 1,
        "response": "2(3x + 1)(3)",
        "hint_level": 0
    })
    assert r1_good.status_code == 200
    assert r1_good.json()["verdict"] == "correct"

    # Step 1: Equivalent simplified form 6(3x + 1) or 18x + 6 also accepted
    r1_equiv = client.post("/api/submit", json={
        "activity_id": act_id,
        "step_index": 1,
        "response": "6(3x + 1)",
        "hint_level": 0
    })
    assert r1_equiv.status_code == 200
    assert r1_equiv.json()["verdict"] == "correct"


def test_tutor_example_and_stuck_endpoints():
    r_ex = client.post("/api/tutor/example", json={"step_index": 1})
    assert r_ex.status_code == 200
    data_ex = r_ex.json()
    assert "title" in data_ex
    assert "solution" in data_ex

    r_stuck = client.post("/api/tutor/stuck", json={"step_index": 1})
    assert r_stuck.status_code == 200
    data_stuck = r_stuck.json()
    assert "reply" in data_stuck
    assert "chain rule" in data_stuck["reply"].lower()


def test_instructor_review_request(chain_activity):
    act_id = chain_activity

    r = client.post("/api/review-request", json={"activity_id": act_id})
    assert r.status_code == 200
    assert r.json()["status"] == "submitted"
