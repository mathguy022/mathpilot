"""One student's work must never surface on another student's page.

Before identity was threaded through the routes, every attempt, submission and
tutor session was written as the first roster student, so this file would fail
end to end. It is the acceptance item "no student can access another student's
private attempts or submissions".
"""
import time

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app import auth
from app.db import (PracticeProgress, Publication, SessionLocal, Submission,
                    TutorEvent, TutorRateHit, TutorSession, TutorThread, User)
from app.main import FLOAT_TUTOR_RATE_LIMIT, app
from tests._identity import new_teacher_client, sign_out

client = TestClient(app)          # reads as student A
other = TestClient(app)           # reads as student B
teacher = new_teacher_client()

ITEM = {"item_id": "q1", "prompt_latex": "Solve $x+1=2$.",
        "steps": [{"step_id": "s1", "description": "Subtract 1.",
                   "expected_input_latex": "x=1", "accepted_equivalents": [],
                   "hints": [], "objective_tags": []}],
        "hints": ["What undoes +1?"], "objective_tags": [], "difficulty": 1,
        "verification": "verified"}

LESSON = {"title": "Isolation lesson", "course": "Calculus 1",
          "topic": "Limits", "objectives": ["Evaluate a limit"],
          "sections": [{"kind": "explanation", "heading": "Idea",
                        "body_markdown": "A limit is $\\lim_{x \\to 1} x$.",
                        "math_latex": [], "examples": []},
                       {"kind": "practice", "heading": "Try",
                        "body_markdown": "Evaluate.",
                        "math_latex": ["\\lim_{x \\to 1} x"],
                        "examples": []}]}


def _two_students():
    db = SessionLocal()
    try:
        rows = [(u.id, u.name) for u in db.query(User)
                .filter(User.role == "student").order_by(User.id)]
    finally:
        db.close()
    if len(rows) < 2:
        pytest.skip("the demo roster needs two students")
    return rows[0][0], rows[1][0]


def _as(c, user_id):
    c.cookies.clear()
    c.cookies.set(auth.COOKIE, auth.issue(user_id))


def _wipe(link, ids):
    db = SessionLocal()
    try:
        pub = db.query(Publication).filter(
            Publication.link_id == link).first()
        if pub:
            db.query(Submission).filter(
                Submission.publication_id == pub.id,
                Submission.user_id.in_(ids)).delete(
                    synchronize_session=False)
        db.query(PracticeProgress).filter(
            PracticeProgress.package_ref == link,
            PracticeProgress.user_id.in_(ids)).delete(synchronize_session=False)
        # Tutor threads are keyed by student, so a student's own rows are the
        # only ones these tests clear.
        for uid in ids:
            db.query(TutorThread).filter(
                TutorThread.key.endswith(f":u{uid}")).delete(
                    synchronize_session=False)
            db.query(TutorRateHit).filter(
                TutorRateHit.student_id == uid).delete(
                    synchronize_session=False)
        sess = [s.id for s in db.query(TutorSession).filter(
            TutorSession.student_id.in_(ids))]
        if sess:
            db.query(TutorEvent).filter(
                TutorEvent.session_id.in_(sess)).delete(
                    synchronize_session=False)
            db.query(TutorSession).filter(
                TutorSession.id.in_(sess)).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


@pytest.fixture(scope="module")
def deck():
    """One published package holding both a lesson and an assignment."""
    lid = teacher.post("/api/lessons/save", json={
        "course_id": 6, "topic": "Limits", "lesson": LESSON,
        "settings": {}}).json()["id"]
    teacher.post(f"/api/lessons/{lid}/approve", json={"lesson": LESSON})
    aid = teacher.post("/api/assignments/save", json={
        "course_id": 6, "topic": "Algebra", "title": "Isolation HW",
        "items": [ITEM], "settings": {}}).json()["id"]
    teacher.post(f"/api/assignments/{aid}/approve", json={})
    p = teacher.post("/api/publications", json={
        "lesson_id": lid, "assignment_id": aid, "title": "Isolation pkg"})
    assert p.status_code == 200, p.text
    link = p.json()["link_id"]
    yield {"link": link, "lesson_id": lid, "assignment_id": aid}
    teacher.delete(f"/api/publications/{link}")
    teacher.delete(f"/api/assignments/{aid}")
    teacher.delete(f"/api/lessons/{lid}")


@pytest.fixture
def pair(deck):
    """Two signed-in students with an empty workspace each."""
    a, b = _two_students()
    _as(client, a)
    _as(other, b)
    _wipe(deck["link"], [a, b])
    yield a, b, deck["link"]
    _wipe(deck["link"], [a, b])
    sign_out(client)
    sign_out(other)


def test_assignment_draft_is_private(pair):
    a, b, link = pair
    marker = "MARK-ALPHA-ANSWER"
    r = client.post("/api/assignments/progress",
                    json={"link_id": link, "answers": {"q1": marker}})
    assert r.status_code == 200, r.text

    assert marker in client.get(f"/myassignments/{link}").text
    theirs = other.get(f"/myassignments/{link}").text
    assert marker not in theirs, "student B saw student A's saved answer"


def test_submission_status_is_private(pair):
    a, b, link = pair
    assert client.post("/api/assignments/submit",
                       json={"link_id": link,
                             "answers": {"q1": "x=1"}}).status_code == 200

    assert "Submitted" in client.get("/assignments").text
    theirs = other.get("/assignments").text
    assert "Submitted" not in theirs, "B's tab showed A's submission"
    assert "Not Started" in theirs


def test_submits_are_not_mutual_locks(pair):
    """A's final submission must not stop B from submitting the same work."""
    a, b, link = pair
    assert client.post("/api/assignments/submit",
                       json={"link_id": link,
                             "answers": {"q1": "x=1"}}).status_code == 200
    r = other.post("/api/assignments/submit",
                   json={"link_id": link, "answers": {"q1": "x=1"}})
    assert r.status_code == 200, r.text


def test_check_budget_is_per_student(pair):
    a, b, link = pair
    for _ in range(3):
        assert client.post("/api/assignments/check", json={
            "link_id": link, "item_id": "q1", "answer": "x=99"}
        ).status_code == 200
    locked = client.post("/api/assignments/check", json={
        "link_id": link, "item_id": "q1", "answer": "x=99"})
    assert locked.json().get("locked") is True, locked.text
    fresh = other.post("/api/assignments/check", json={
        "link_id": link, "item_id": "q1", "answer": "x=99"})
    assert fresh.status_code == 200
    assert fresh.json().get("tries") == 1, fresh.text


def test_practice_progress_is_private(pair):
    a, b, link = pair
    assert client.post("/api/practice/progress", json={
        "ref": link, "item_id": "practice-1", "status": "solved",
        "draft": "ALPHA-DRAFT"}).status_code == 200

    mine = client.post("/api/practice/items", json={"ref": link}).json()
    assert mine["progress"]["practice-1"]["status"] == "solved"
    theirs = other.post("/api/practice/items", json={"ref": link}).json()
    assert theirs["progress"] == {}, "B inherited A's practice progress"


def test_tutor_session_is_private(pair):
    a, b, link = pair
    db = SessionLocal()
    try:
        s = TutorSession(student_id=a, course_id="default",
                         problem_text="10 = 3x + 1", status="active",
                         total_steps=2, current_step_index=0,
                         solution_plan_json="[]")
        db.add(s)
        db.flush()
        db.add(TutorEvent(session_id=s.id, step_index=0,
                          student_input="MARK-ALPHA-ATTEMPT",
                          verdict="incorrect", event_type="message",
                          tutor_reply="What do you get if you undo the +1?"))
        db.commit()
    finally:
        db.close()

    mine = client.get("/api/tutor/session?course=default").json()
    theirs = other.get("/api/tutor/session?course=default").json()
    assert mine.get("active") is True, mine
    assert "MARK-ALPHA-ATTEMPT" in str(mine.get("messages"))
    assert theirs.get("active") is False, theirs
    assert theirs.get("messages") == [], \
        "B resumed A's private tutoring transcript"


def test_hint_ladder_is_per_student(pair):
    """Two students on the same item climb two separate ladders.

    The rung each reached used to live in one process dict keyed by item
    alone, so the second student inherited the first one's hints.
    """
    a, b, link = pair

    def ask(c):
        return c.post("/api/practice/attempt", json={
            "ref": link, "item_id": "practice-1",
            "response": "I'm stuck — give me a hint"})

    with patch("app.config.ai_configured", return_value=True), \
         patch("app.nebius.float_chat", return_value=None), \
         patch("app.nebius.generate_plan", return_value=None), \
         patch("app.nebius.normalize_via_llm", return_value=None), \
         patch("app.nebius.judge_conceptual", return_value="unclear"):
        assert ask(client).json()["hintLevel"] == 1
        assert ask(other).json()["hintLevel"] == 1, \
            "B started on the rung A had already spent"
        assert ask(client).json()["hintLevel"] == 2

    db = SessionLocal()
    try:
        stored = {t.key: t.hints for t in db.query(TutorThread).filter(
            TutorThread.key.startswith(f"practice:{link}:practice-1:"))}
    finally:
        db.close()
    assert stored.get(f"practice:{link}:practice-1:u{a}") == 2, stored
    assert stored.get(f"practice:{link}:practice-1:u{b}") == 1, stored


def test_tutor_rate_window_is_per_student(pair):
    """A's messages cannot use up B's allowance."""
    a, b, link = pair
    db = SessionLocal()
    try:
        db.add_all([TutorRateHit(student_id=a, ts=time.time())
                    for _ in range(FLOAT_TUTOR_RATE_LIMIT)])
        db.commit()
    finally:
        db.close()

    def ask(c):
        return c.post("/api/tutor/message", json={
            "message": "10 = 3x + 1", "pageContext": {"type": "general"}})

    assert ask(client).status_code == 429
    assert ask(other).status_code == 200
