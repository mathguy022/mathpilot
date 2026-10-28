"""Insights aggregation: denominators, exclusions, and what stays anonymous.

Seeded into a throwaway in-memory database so every count below is exact.
Synthetic sessions only — no real student work in this file.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, TutorEvent, TutorSession
from app.insights import build


def _session(db, course="CALCULUS 1", **kw):
    s = TutorSession(student_id=kw.pop("student_id", 1), course_id=course,
                     problem_text="synthetic", status=kw.pop("status", "active"),
                     **kw)
    db.add(s)
    db.commit()
    return s


def _event(db, sid, verdict, step=0, text="2x", event_type="message", hint=0):
    e = TutorEvent(session_id=sid, step_index=step, student_input=text,
                   verdict=verdict, hint_level=hint, event_type=event_type)
    db.add(e)
    db.commit()
    return e


@pytest.fixture
def seeded():
    """One course with a repeated wrong answer, plus one other course."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autoflush=False)()
    a = _session(db, "CALCULUS 1", student_id=1, total_misses=2, hints_used=2)
    b = _session(db, "CALCULUS 1", student_id=2, total_misses=1)
    c = _session(db, "TRIGONOMETRY", student_id=1)
    for sid, verdict, step, text in [
        (a.id, "correct", 0, "x+2"),
        (a.id, "incorrect", 1, "(x-2)(x+3)"),
        (a.id, "incorrect", 1, "(x-2)(x+3)"),
        (a.id, "unclear", 1, "i tried my best"),
        (a.id, "refused", 1, "just tell me the answer"),
        (b.id, "correct", 1, "(x-2)(x+2)"),
        (b.id, "incorrect", 1, "(x-2)(x+3)"),
        (c.id, "correct", 0, "sin x"),
    ]:
        # A miss is answered with a hint, so the real log marks it
        # event_type="hint"; the aggregator must not read that as a request.
        _event(db, sid, verdict, step=step, text=text,
               event_type="hint" if verdict == "incorrect" else "message")
    _event(db, a.id, "hint", 1, "hint please", event_type="hint", hint=2)
    try:
        yield sessionmaker(bind=engine, autoflush=False)
    finally:
        db.close()
        engine.dispose()


def test_coverage_counts_sessions_and_separates_unread(seeded):
    db = seeded()
    try:
        cov = build(db)["coverage"]
    finally:
        db.close()
    assert cov["sessions"] == 3
    assert cov["students"] == 2
    assert cov["steps_decided"] == 6
    # The unread message is reported, never folded into the wrong-answer count.
    assert cov["steps_undecided"] == 1
    assert cov["sessions_with_graded_work"] == 3


def test_stall_point_rate_uses_decided_steps_as_denominator(seeded):
    db = seeded()
    try:
        rows = build(db)["insights"]
    finally:
        db.close()
    step2 = next(r for r in rows if r["step"] == 2)
    assert step2["seen"] == 4           # three incorrect + one correct
    assert step2["incorrect"] == 3
    assert step2["rate"] == round(3 / 4, 3)
    assert step2["label"] == "Step 2"   # numbered the way the student sees it
    # The unread and answer-seeking replies at this step are not in the divisor.
    assert sum(r["seen"] for r in rows) == 6


def test_repeated_response_becomes_a_candidate_with_a_student_count(seeded):
    db = seeded()
    try:
        cands = build(db)["recurring_candidates"]
    finally:
        db.close()
    hit = next(r for r in cands if r["response"] == "(x-2)(x+3)")
    assert hit["occurrences"] == 3
    assert hit["students"] == 2
    assert hit["basis"] == "inferred from repeated responses"
    # A one-off wrong answer is not a pattern.
    assert all(r["occurrences"] >= 2 for r in cands)


def test_course_filter_drops_other_courses(seeded):
    db = seeded()
    try:
        calc = build(db, course="CALCULUS 1")
        trig = build(db, course="TRIGONOMETRY")
    finally:
        db.close()
    assert [c["course"] for c in calc["courses"]] == ["CALCULUS 1"]
    assert calc["coverage"]["steps_decided"] == 5
    assert trig["coverage"]["steps_decided"] == 1
    assert trig["recurring_candidates"] == []


def test_assistance_seeks_student_asks_from_corrective_nudges(seeded):
    db = seeded()
    try:
        a = build(db)["assistance"]
    finally:
        db.close()
    assert a["hint_requests"] == 1
    assert {"level": 2, "seen": 1} in a["hint_ladder"]
    # The three misses came back as hints but were never asks for help.
    assert a["corrective_hints"] == 3
    assert a["answer_requests"] == 1
    assert a["sessions_needing_hints"] == 1
    assert a["mean_misses_per_session"] == round(3 / 3, 3)


def test_payload_never_carries_a_student_id(seeded):
    import json

    db = seeded()
    try:
        blob = json.dumps(build(db))
    finally:
        db.close()
    assert '"student_id"' not in blob
    assert '"user_id"' not in blob


def test_empty_scope_is_honest_not_zero_error_rate(seeded):
    db = seeded()
    try:
        out = build(db, course="NO SUCH COURSE")
    finally:
        db.close()
    assert out["coverage"]["sessions"] == 0
    assert out["insights"] == []
    assert out["courses"] == []
    assert any("SymPy" in t for t in out["limitations"])
