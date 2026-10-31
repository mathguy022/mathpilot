"""Tests for Public Demo Access: visitors default to Dr. Sarah Wilson.

Ensures any visitor to the deployed MathPilot website can explore the
application (including instructor builder, assignments, drafts, and AI tutor)
without requiring personal credentials or manual 'View as' switching.
"""
from fastapi.testclient import TestClient

from app import auth
from app.db import SessionLocal, User
from app.main import app

client = TestClient(app)


def test_public_demo_visitor_defaults_to_dr_sarah_wilson(monkeypatch):
    monkeypatch.setenv("MATHPILOT_PUBLIC_DEMO", "true")
    client.cookies.clear()

    # 1. Root and student workspace render with Dr. Sarah Wilson context
    r = client.get("/")
    assert r.status_code == 200
    assert "Dr. Sarah Wilson" in r.text
    assert "DW" in r.text or "SW" in r.text

    # 2. Instructor dashboard is directly open without manual sign in
    dash = client.get("/instructor/dashboard")
    assert dash.status_code == 200
    assert "Workspace Canvas" in dash.text
    assert "Generate Lesson" in dash.text
    assert "Create Assignment" in dash.text

    # 3. Class analytics / insights are accessible
    analytics = client.get("/api/analytics")
    assert analytics.status_code == 200
    assert "insights" in analytics.json()

    # 4. Drafts page is accessible
    drafts = client.get("/instructor/drafts")
    assert drafts.status_code == 200


def test_public_demo_visitor_can_switch_to_student(monkeypatch):
    monkeypatch.setenv("MATHPILOT_PUBLIC_DEMO", "true")
    client.cookies.clear()

    db = SessionLocal()
    student = db.query(User).filter(User.role == "student").first()
    db.close()
    assert student is not None

    # Switch to student via /view-as
    r = client.post("/view-as", data={"user_id": student.id, "back": "/"}, follow_redirects=True)
    assert r.status_code == 200
    assert student.name in r.text

    # As a student, instructor endpoints must return 403
    assert client.get("/instructor/dashboard").status_code == 403
    assert client.get("/api/analytics").status_code == 403


def test_public_demo_visitor_can_explicitly_sign_out(monkeypatch):
    monkeypatch.setenv("MATHPILOT_PUBLIC_DEMO", "true")
    client.cookies.clear()

    # Initial visitor has default access
    assert client.get("/instructor/dashboard").status_code == 200

    # User clicks 'Sign out' (user_id = 0)
    r = client.post("/view-as", data={"user_id": 0, "back": "/"}, follow_redirects=True)
    assert r.status_code == 200

    # Now explicitly signed out -> instructor endpoints return 403
    assert client.get("/instructor/dashboard").status_code == 403
    assert client.get("/api/analytics").status_code == 403


def test_public_demo_visitor_switch_back_to_instructor(monkeypatch):
    monkeypatch.setenv("MATHPILOT_PUBLIC_DEMO", "true")
    client.cookies.clear()

    db = SessionLocal()
    teacher = db.query(User).filter(User.role == "instructor").first()
    db.close()

    # Switch to teacher explicitly
    r = client.post("/view-as", data={"user_id": teacher.id, "back": "/instructor/dashboard"}, follow_redirects=True)
    assert r.status_code == 200
    assert teacher.name in r.text
