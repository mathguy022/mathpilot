"""Tests for the saved-drafts manager page and draft delete endpoints."""
import os

os.environ["DEMO_MODE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

LESSON = {"title": "Drafts Test Lesson", "course": "Calculus I",
          "topic": "Limits", "objectives": ["Evaluate simple limits"],
          "sections": []}
ASSIGN_ITEM = {"item_id": "q1",
               "prompt_latex": "Solve $x+1=2$.",
               "steps": []}


def test_drafts_page_renders():
    r = client.get("/instructor/drafts")
    assert r.status_code == 200
    assert "Saved Drafts" in r.text
    assert "📖 Lessons" in r.text
    assert "📝 Assignments" in r.text
    assert "Back to Builder" in r.text


def test_lesson_draft_delete_lifecycle():
    r = client.post("/api/lessons/save", json={
        "lesson": LESSON, "topic": "Limits", "settings": {}})
    assert r.status_code == 200, r.text
    lid = r.json()["id"]
    try:
        page = client.get("/instructor/drafts")
        assert LESSON["title"] in page.text
        d = client.delete(f"/api/lessons/{lid}")
        assert d.status_code == 200
        assert d.json() == {"deleted": lid}
        assert client.get(f"/api/lessons/{lid}").status_code == 404
        assert client.delete(f"/api/lessons/{lid}").status_code == 404
    finally:
        client.delete(f"/api/lessons/{lid}")


def test_lesson_delete_blocked_by_publication():
    r = client.post("/api/lessons/save", json={
        "lesson": dict(LESSON, title="Drafts Locked Lesson"),
        "topic": "Limits", "settings": {}})
    lid = r.json()["id"]
    a = client.post(f"/api/lessons/{lid}/approve", json={})
    assert a.status_code == 200, a.text
    p = client.post("/api/publications",
                    json={"lesson_id": lid, "title": "Locked pkg"})
    assert p.status_code == 200, p.text
    link_id = p.json()["link_id"]
    try:
        d = client.delete(f"/api/lessons/{lid}")
        assert d.status_code == 409
        assert "student link" in d.json()["detail"]
        page = client.get("/instructor/drafts")
        assert f"/learn/{link_id}" in page.text
    finally:
        client.delete(f"/api/lessons/{lid}")


def test_assignment_draft_delete_lifecycle():
    r = client.post("/api/assignments/save", json={
        "title": "Drafts Test Assignment", "topic": "Algebra",
        "items": [ASSIGN_ITEM], "settings": {}})
    assert r.status_code == 200, r.text
    aid = r.json()["id"]
    try:
        page = client.get("/instructor/drafts")
        assert "Drafts Test Assignment" in page.text
        d = client.delete(f"/api/assignments/{aid}")
        assert d.status_code == 200
        assert d.json() == {"deleted": aid}
        assert client.get(f"/api/assignments/{aid}").status_code == 404
        assert client.delete(f"/api/assignments/{aid}").status_code == 404
    finally:
        client.delete(f"/api/assignments/{aid}")


def test_assignment_delete_blocked_by_publication():
    r = client.post("/api/assignments/save", json={
        "title": "Drafts Locked Assignment", "topic": "Algebra",
        "items": [ASSIGN_ITEM], "settings": {}})
    aid = r.json()["id"]
    a = client.post(f"/api/assignments/{aid}/approve", json={})
    assert a.status_code == 200, a.text
    p = client.post("/api/publications",
                    json={"assignment_id": aid, "title": "Locked pkg A"})
    assert p.status_code == 200, p.text
    link_id = p.json()["link_id"]
    try:
        d = client.delete(f"/api/assignments/{aid}")
        assert d.status_code == 409
        assert "student link" in d.json()["detail"]
        page = client.get("/instructor/drafts")
        assert f"/learn/{link_id}" in page.text
    finally:
        client.delete(f"/api/assignments/{aid}")


def test_retire_link_reverts_content_to_approved_then_deletable():
    r = client.post("/api/lessons/save", json={
        "lesson": dict(LESSON, title="Drafts Retire Lesson"),
        "topic": "Limits", "settings": {}})
    lid = r.json()["id"]
    assert client.post(f"/api/lessons/{lid}/approve", json={}).status_code == 200
    p = client.post("/api/publications",
                    json={"lesson_id": lid, "title": "Retire pkg"})
    assert p.status_code == 200, p.text
    link_id = p.json()["link_id"]
    try:
        assert client.get(f"/learn/{link_id}").status_code == 200
        d = client.delete(f"/api/publications/{link_id}")
        assert d.status_code == 200
        assert d.json() == {"retired": link_id, "reverted": ["lesson"]}
        assert client.get(f"/learn/{link_id}").status_code == 404
        got = client.get(f"/api/lessons/{lid}")
        assert got.status_code == 200
        assert got.json()["status"] == "approved"
        assert client.delete(f"/api/lessons/{lid}").status_code == 200
        assert client.get(f"/api/lessons/{lid}").status_code == 404
    finally:
        client.delete(f"/api/publications/{link_id}")
        client.delete(f"/api/lessons/{lid}")


def test_retire_unknown_link_404():
    assert client.delete("/api/publications/nonexistent").status_code == 404


def test_retire_one_link_keeps_other_content_published():
    r = client.post("/api/lessons/save", json={
        "lesson": dict(LESSON, title="Drafts Combo Lesson"),
        "topic": "Limits", "settings": {}})
    lid = r.json()["id"]
    assert client.post(f"/api/lessons/{lid}/approve", json={}).status_code == 200
    a = client.post("/api/assignments/save", json={
        "title": "Drafts Combo Assignment", "topic": "Algebra",
        "items": [ASSIGN_ITEM], "settings": {}})
    aid = a.json()["id"]
    assert client.post(f"/api/assignments/{aid}/approve", json={}).status_code == 200
    p = client.post("/api/publications", json={
        "lesson_id": lid, "assignment_id": aid, "title": "Combo pkg"})
    assert p.status_code == 200, p.text
    link_id = p.json()["link_id"]
    try:
        d = client.delete(f"/api/publications/{link_id}")
        assert d.status_code == 200
        assert d.json() == {"retired": link_id,
                            "reverted": ["lesson", "assignment"]}
        assert client.get(f"/api/lessons/{lid}").json()["status"] == "approved"
        assert client.get(f"/api/assignments/{aid}").json()["status"] == "approved"
        assert client.delete(f"/api/lessons/{lid}").status_code == 200
        assert client.delete(f"/api/assignments/{aid}").status_code == 200
    finally:
        client.delete(f"/api/publications/{link_id}")
        client.delete(f"/api/lessons/{lid}")
        client.delete(f"/api/assignments/{aid}")
