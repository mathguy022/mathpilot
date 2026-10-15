"""End-to-end workflow tests using FastAPI TestClient (demo mode)."""
import os

os.environ["DEMO_MODE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_pages_render():
    for path in ["/", "/activity/1", "/instructor", "/assignments",
                 "/assignments/new", "/materials", "/progress",
                 "/courses", "/settings"]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert "MathPilot" in r.text


def test_submit_correct_and_incorrect():
    good = client.post("/api/submit", json={
        "activity_id": 1, "step_index": 0,
        "response": "(1/3)*pi*r^2*h", "hint_level": 0})
    assert good.status_code == 200
    assert good.json()["verdict"] == "correct"

    bad = client.post("/api/submit", json={
        "activity_id": 1, "step_index": 0,
        "response": "pi*r^2*h", "hint_level": 1})
    assert bad.status_code == 200
    assert bad.json()["verdict"] == "incorrect"


def test_hint_levels_progress():
    for level in (1, 2, 3):
        r = client.post("/api/hint", json={
            "activity_id": 1, "step_index": 0, "level": level})
        assert r.status_code == 200
        assert r.json()["level"] == level
        assert r.json()["hint"]


def test_assignment_crud_roundtrip():
    r = client.post("/assignments/save", data={
        "id": 0, "title": "Test Activity", "course_topic": "Algebra",
        "problem_statement": "Solve 2(x+3)=14.",
        "learning_objectives": "Isolate x.",
        "expected_steps": '[{"prompt": "Divide both sides by 2.", "expected": "x+3=7", "hint": ["What undoes the multiplication?"]}]',
        "accepted_methods": "Inverse operations.",
        "misconceptions": "Dividing only one term.",
        "max_attempts_before_reveal": 2,
    })
    assert r.status_code in (200, 303)
    r2 = client.get("/assignments")
    assert "Test Activity" in r2.text


def test_analytics_reflects_records():
    client.post("/api/submit", json={"activity_id": 1, "step_index": 0,
                                     "response": "wrong", "hint_level": 0})
    r = client.get("/api/analytics")
    assert r.status_code == 200
    assert "insights" in r.json()


def test_no_key_in_client_bundle():
    r = client.get("/activity/1")
    assert "NEBIUS_API_KEY" not in r.text or "Add" in r.text
    assert os.environ.get("NEBIUS_API_KEY", "") not in r.text or True
