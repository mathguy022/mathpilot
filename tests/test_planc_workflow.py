"""End-to-end Plan C workflow: draft -> approve -> publish -> student view.

Uses synthetic lesson/assignment payloads (no AI calls). Verifies:
draft persistence, approval derivation, frozen publishing, and that
student endpoints never expose instructor solutions.
"""
import os

os.environ["DEMO_MODE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

LESSON = {
    "title": "Limits intro", "course": "Calculus I", "topic": "Limits",
    "objectives": ["Evaluate simple limits"],
    "prerequisites": ["Algebra"],
    "sections": [
        {"kind": "explanation", "heading": "Idea",
         "body_markdown": "A limit is $\\lim_{x \\to 1} x$.",
         "math_latex": [], "examples": []},
        {"kind": "practice", "heading": "Try",
         "body_markdown": "Evaluate.",
         "math_latex": ["\\lim_{x \\to 1} x"], "examples": []},
        {"kind": "solutions", "heading": "Keys",
         "body_markdown": "Answer: $1$.",
         "math_latex": [], "examples": []},
    ],
}


def test_lesson_draft_approve_publish_learn():
    # Save draft.
    r = client.post("/api/lessons/save", json={
        "course_id": 1, "topic": "Limits", "lesson": LESSON, "settings": {}})
    assert r.status_code == 200, r.text
    lid = r.json()["id"]
    assert r.json()["status"] == "draft"
    # Draft survives a re-fetch (persistence, not in-memory).
    g = client.get(f"/api/lessons/{lid}")
    assert g.status_code == 200
    assert g.json()["lesson"]["title"] == "Limits intro"
    # Approve: derives student content, strips solutions.
    a = client.post(f"/api/lessons/{lid}/approve", json={"lesson": LESSON})
    assert a.status_code == 200, a.text
    assert a.json()["status"] == "approved"
    assert a.json()["guided_items"] == 1
    # Publish lesson alone.
    p = client.post("/api/publications", json={
        "lesson_id": lid, "title": "Week 1 — Limits"})
    assert p.status_code == 200, p.text
    link = p.json()["link_id"]
    # Student page: no solutions, no answer keys.
    page = client.get(f"/learn/{link}")
    assert page.status_code == 200
    assert "Limits intro" in page.text
    assert "Answer: $1$" not in page.text
    assert "Keys" not in page.text
    api = client.get(f"/api/learn/{link}")
    assert api.status_code == 200
    assert "solutions" not in str(api.json()).lower()
    # Published lesson is frozen.
    again = client.post("/api/lessons/save", json={
        "id": lid, "topic": "Limits", "lesson": LESSON, "settings": {}})
    assert again.status_code == 400


def test_assignment_save_approve_publish():
    items = [{
        "item_id": "q1", "prompt_latex": "Solve $x+1=2$.",
        "steps": [{"step_id": "s1", "description": "Subtract 1.",
                   "expected_input_latex": "x=1",
                   "accepted_equivalents": [], "hints": [],
                   "objective_tags": []}],
        "hints": ["What undoes +1?"], "objective_tags": [],
        "difficulty": 1, "verification": "needs-review",
    }]
    solutions = {"q1": {"final_answer_latex": "x=1",
                        "solution_steps": ["x=1"], "explanation": "t"}}
    r = client.post("/api/assignments/save", json={
        "course_id": 1, "topic": "Algebra", "title": "HW 1",
        "items": items, "solutions": solutions, "settings": {}})
    assert r.status_code == 200, r.text
    aid = r.json()["id"]
    g = client.get(f"/api/assignments/{aid}")
    assert g.status_code == 200
    assert g.json()["items"][0]["verification"] == "verified"
    ap = client.post(f"/api/assignments/{aid}/approve", json={})
    assert ap.status_code == 200
    p = client.post("/api/publications", json={"assignment_id": aid})
    assert p.status_code == 200, p.text
    page = client.get(f"/learn/{p.json()['link_id']}")
    assert page.status_code == 200
    # Solutions never reach the student page.
    assert "solution_steps" not in page.text


def test_publish_requires_approval():
    r = client.post("/api/lessons/save", json={
        "topic": "T", "lesson": dict(LESSON, title="Unapproved"),
        "settings": {}})
    lid = r.json()["id"]
    p = client.post("/api/publications", json={"lesson_id": lid})
    assert p.status_code == 400
    # Assign also requires approval, and requires a due date.
    assert client.post("/api/assign", json={
        "lesson_id": lid, "due": "2026-12-01"}).status_code == 400
    appr = client.post(f"/api/lessons/{lid}/approve",
                       json={"lesson": dict(LESSON, title="Unapproved")})
    assert appr.status_code == 200
    nodue = client.post("/api/assign", json={"lesson_id": lid})
    assert nodue.status_code == 400
    a = client.post("/api/assign", json={
        "lesson_id": lid, "course": "Calculus 1", "due": "2026-12-01"})
    assert a.status_code == 200, a.text
    assert a.json()["due"] == "2026-12-01"
    assert a.json()["course"] == "Calculus 1"
    page = client.get(f"/learn/{a.json()['link_id']}")
    assert page.status_code == 200


def test_bad_links_404():
    assert client.get("/learn/deadbeef").status_code == 404
    assert client.get("/api/learn/deadbeef").status_code == 404
