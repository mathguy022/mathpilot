"""The role boundary in code: signed identity, instructor-only insights.

Roster accounts are the synthetic demo users, and the attempt rows below are
marked with invented text so a test can tell whose page it is reading.
"""
import pytest
from fastapi.testclient import TestClient

from app import auth
from app.db import Activity, Attempt, SessionLocal, User
from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def roster():
    db = SessionLocal()
    try:
        users = [(u.id, u.name, u.role) for u in db.query(User).order_by(User.id)]
    finally:
        db.close()
    students = [u for u in users if u[2] == "student"]
    teachers = [u for u in users if u[2] == "instructor"]
    if len(students) < 2 or not teachers:
        pytest.skip("the demo roster needs two students and one instructor")
    return {"students": students, "teachers": teachers}


@pytest.fixture
def marked_attempts(roster):
    """Two students, one attempt each, with text only its own page may show."""
    a_id = roster["students"][0][0]
    b_id = roster["students"][1][0]
    db = SessionLocal()
    try:
        act = Activity(title="identity probe", problem_statement="x + 1 = 2",
                       expected_steps_json="[]", hint_sequence_json="[]")
        db.add(act)
        db.commit()
        db.add_all([
            Attempt(activity_id=act.id, user_id=a_id, step_index=0,
                    student_response="MARK-ALPHA", verdict="correct"),
            Attempt(activity_id=act.id, user_id=b_id, step_index=0,
                    student_response="MARK-BETA", verdict="correct"),
        ])
        db.commit()
        yield a_id, b_id
    finally:
        db.query(Attempt).filter(Attempt.activity_id == act.id).delete()
        db.query(Activity).filter(Activity.id == act.id).delete()
        db.commit()
        db.close()


def _as(user_id):
    client.cookies.clear()
    if user_id is not None:
        client.cookies.set(auth.COOKIE, auth.issue(user_id))


def test_token_round_trip_and_rejects_tampering():
    token = auth.issue(7)
    assert auth.resolve(token) == 7
    body, _, sig = token.rpartition(".")
    assert auth.resolve(f'{"x" + body[1:]}.{sig}') is None   # payload edited
    # Flip the last hex digit to a different digit, never to itself.
    sig_bad = sig[:-1] + ("a" if sig[-1] != "a" else "b")
    assert auth.resolve(f"{body}.{sig_bad}") is None         # signature edited
    for bad in (None, "", "7", "garbage", "a.b", "3."):
        assert auth.resolve(bad) is None


def test_insights_need_an_identity_at_all():
    _as(None)
    assert client.get("/api/analytics").status_code == 403


def test_a_student_identity_is_refused_by_name(roster):
    student = roster["students"][0]
    _as(student[0])
    try:
        r = client.get("/api/analytics")
        assert r.status_code == 403
        assert student[1] in r.json()["detail"]
        assert "student" in r.json()["detail"]
    finally:
        client.cookies.clear()


def test_a_bare_user_id_is_not_an_identity(roster):
    """A student cannot become the teacher by typing the cookie by hand."""
    teacher = roster["teachers"][0][0]
    client.cookies.clear()
    client.cookies.set(auth.COOKIE, str(teacher))
    try:
        assert client.get("/api/analytics").status_code == 403
    finally:
        client.cookies.clear()


def test_an_instructor_identity_can_read_insights(roster):
    _as(roster["teachers"][0][0])
    try:
        r = client.get("/api/analytics")
        assert r.status_code == 200
        assert {"coverage", "insights", "recurring_candidates",
                "limitations"} <= set(r.json())
    finally:
        client.cookies.clear()


def test_view_as_signs_in_and_out(roster):
    teacher = roster["teachers"][0]
    client.cookies.clear()
    r = client.post("/view-as", data={"user_id": teacher[0], "back": "/instructor"})
    assert r.status_code == 200          # followed the redirect to the page
    assert teacher[1] in r.text          # header now names that account
    assert client.get("/api/analytics").status_code == 200

    client.post("/view-as", data={"user_id": 0, "back": "/"})
    assert client.get("/api/analytics").status_code == 403
    client.cookies.clear()


def test_view_as_rejects_an_unknown_account():
    client.cookies.clear()
    assert client.post("/view-as", data={"user_id": 99999,
                                         "back": "/"}).status_code == 404


def test_identity_cookie_carries_its_flags(roster):
    """HttpOnly + Lax always; Secure only when the deployment asks for it."""
    client.cookies.clear()
    r = client.post("/view-as", data={"user_id": roster["teachers"][0][0],
                                      "back": "/"}, follow_redirects=False)
    cookie = r.headers["set-cookie"]
    assert auth.COOKIE + "=" in cookie
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert "Secure" not in cookie          # the demo runs on plain http
    client.cookies.clear()


def test_view_as_refuses_a_foreign_return_path(roster):
    client.cookies.clear()
    r = client.post("/view-as", data={"user_id": roster["teachers"][0][0],
                                      "back": "//evil.example/steal"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    client.cookies.clear()


def test_progress_shows_only_the_signed_in_student(marked_attempts, roster):
    a_id, b_id = marked_attempts
    for uid, own, other in ((a_id, "MARK-ALPHA", "MARK-BETA"),
                            (b_id, "MARK-BETA", "MARK-ALPHA")):
        _as(uid)
        try:
            html = client.get("/progress").text
        finally:
            client.cookies.clear()
        assert own in html
        assert other not in html


def test_progress_without_identity_still_renders():
    _as(None)
    r = client.get("/progress")
    assert r.status_code == 200
    assert "Track Your Progress" in r.text
