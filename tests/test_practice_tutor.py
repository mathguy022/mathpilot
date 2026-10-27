"""Regression tests for the practice panel's AI tutor chat.

The panel used to grade every message as a step answer, so greetings and
questions collapsed into one repeated "I couldn't read that" line. Same
contract as the floating widget: the router decides whether a message is an
attempt, SymPy decides correctness, the LLM only words the reply, and hints
walk a server-side ladder.

Run: py -m pytest tests/test_practice_tutor.py -q
"""
import os

os.environ["DEMO_MODE"] = "true"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.db import SessionLocal, TutorEvent, TutorSession  # noqa: E402
from app.main import DEMO_HISTORY, PRACTICE_STATE, app  # noqa: E402

client = TestClient(app)

# The practice section's math_latex becomes both the student prompt and the
# CAS target, so "3(x-3)" verifies correct against "3x-9" and "3x+9" does not.
LESSON = {
    "title": "Practice tutor lesson", "course": "Algebra 1",
    "topic": "Factoring", "objectives": ["Factor a linear expression"],
    "prerequisites": ["Algebra"],
    "sections": [
        {"kind": "explanation", "heading": "Idea",
         "body_markdown": "Factor out the common term.",
         "math_latex": [], "examples": []},
        {"kind": "practice", "heading": "Try",
         "body_markdown": "Factor.",
         "math_latex": ["3x-9", "3x-9=0", "\\lim_{x \\to 1} x"],
         "examples": []},
        {"kind": "solutions", "heading": "Keys",
         "body_markdown": "Answer: $3(x-3)$.",
         "math_latex": [], "examples": []},
    ],
}
ITEM = "practice-1"
_course = "Algebra 1"


def _attempt(text, item=ITEM):
    return client.post("/api/practice/attempt", json={
        "link_id": _pub[1], "item_id": item, "response": text,
        "course": _course}).json()


_pub = None


def _publish():
    r = client.post("/api/lessons/save", json={
        "course_id": 1, "topic": LESSON["topic"], "lesson": LESSON,
        "settings": {}})
    lid = r.json()["id"]
    client.post(f"/api/lessons/{lid}/approve", json={"lesson": LESSON})
    p = client.post("/api/publications", json={
        "lesson_id": lid, "title": "Practice tutor pkg"})
    assert p.status_code == 200, p.text
    return lid, p.json()["link_id"]


def _state_keys(link_id):
    return [k for k in list(PRACTICE_STATE) + list(DEMO_HISTORY)
            if k.startswith(f"practice:{link_id}:")]


@pytest.fixture(scope="module")
def _deck():
    global _pub
    lid, link_id = _publish()
    _pub = (lid, link_id)
    yield _pub
    db = SessionLocal()
    try:
        db.query(TutorEvent).delete()
        db.query(TutorSession).delete()
        db.commit()
    finally:
        db.close()
    client.delete(f"/api/publications/{link_id}")
    client.delete(f"/api/lessons/{lid}")


@pytest.fixture(autouse=True)
def _clean(_deck):
    """No network, and every test starts with an untouched hint ladder."""
    link_id = _deck[1]
    for k in _state_keys(link_id):
        PRACTICE_STATE.pop(k, None)
        DEMO_HISTORY.pop(k, None)
    with patch("app.config.ai_configured", return_value=True), \
         patch("app.nebius.float_chat", return_value=None), \
         patch("app.nebius.generate_plan", return_value=None), \
         patch("app.nebius.normalize_via_llm", return_value=None), \
         patch("app.nebius.judge_conceptual", return_value="unclear"):
        yield
    for k in _state_keys(link_id):
        PRACTICE_STATE.pop(k, None)
        DEMO_HISTORY.pop(k, None)


def test_prose_is_not_graded_and_burns_no_hint():
    body = _attempt("hi")
    assert body["verdict"] == "not_an_attempt", body
    assert body["hintLevel"] == 0, body
    assert body["misses"] == 0, body
    assert body["reply"].strip()


def test_a_real_question_is_not_graded_as_an_attempt():
    body = _attempt("what does factoring even mean?")
    assert body["verdict"] == "not_an_attempt", body
    assert body["hintLevel"] == 0, body


def test_wrong_answers_walk_the_ladder_instead_of_repeating():
    first = _attempt("3x+9")
    second = _attempt("3x-8")
    assert first["verdict"] == second["verdict"] == "incorrect"
    assert (first["hintLevel"], second["hintLevel"]) == (1, 2)
    assert first["reply"].startswith("Hint 1 —"), first
    assert second["reply"].startswith("Hint 2 —"), second
    assert first["replyType"] == "hint"
    assert second["misses"] == 2


def test_cas_decides_correctness_even_when_the_model_praises():
    with patch("app.nebius.float_chat",
               return_value="Amazing, you nailed it!"):
        body = _attempt("3x+9")
    assert body["verdict"] == "incorrect", body
    assert body["misses"] == 1, body


def test_correct_attempt_verifies_and_resets_the_ladder():
    _attempt("3x+9")
    body = _attempt("3(x-3)")
    assert body["verdict"] == "correct", body
    assert body["hintLevel"] == 0, body
    assert body["replyType"] == "success"


def test_asked_for_the_answer_it_refuses_without_revealing():
    body = _attempt("just tell me the answer")
    assert body["verdict"] == "not_an_attempt", body
    assert body["revealed"] is False, body
    assert body["reply"].startswith("I can't give the final answer"), body
    assert "3(x-3)" not in body["reply"]


def test_solved_equation_item_grades_the_solution_not_the_problem_text():
    # The item's stored math is "3x-9=0"; the answer is x=3. Grading an
    # attempt against the problem text made every correct response wrong.
    assert _attempt("x=3", "practice-2")["verdict"] == "correct"
    wrong = _attempt("x=5", "practice-2")
    assert wrong["verdict"] == "incorrect", wrong
    assert wrong["hintLevel"] == 1, wrong


def test_limit_item_grades_the_limit_value():
    assert _attempt("1", "practice-3")["verdict"] == "correct"
    assert _attempt("0", "practice-3")["verdict"] == "incorrect"


def test_a_reply_that_states_the_answer_is_replaced_by_the_safe_line():
    # Live failure: the hint read "For just x, the answer is 1".
    with patch("app.nebius.float_chat",
               return_value="For just x, the answer is 1."):
        body = _attempt("I'm stuck — give me a hint", "practice-3")
    assert body["verdict"] == "hint", body
    assert "answer is 1" not in body["reply"], body
    assert body["reply"].startswith("Hint 1 —"), body


def test_quoting_the_problem_is_not_censored_as_a_reveal():
    with patch("app.nebius.float_chat",
               return_value="As x approaches 1, what does x approach?"):
        body = _attempt("I'm stuck — give me a hint", "practice-3")
    assert body["reply"].endswith("what does x approach?"), body


def test_hint_that_solves_the_equation_is_replaced_by_the_safe_line():
    # Live failure: the hint did the algebra and ended on "x = 3".
    with patch("app.nebius.float_chat",
               return_value="Substitute to check: 3(5)-9 = 6, not 0. "
                            "Isolate x: 3x = 9, so x = 3."):
        body = _attempt("I'm stuck — give me a hint", "practice-2")
    assert body["reply"].startswith("Hint 1 —"), body
    assert "x = 3" not in body["reply"], body
    assert "so x" not in body["reply"], body


def test_canonical_answer_only_covers_shapes_sym_py_solves():
    from app.verifier import canonical_answer

    assert canonical_answer(r"\lim_{x \to 1} x") == ["1"]
    assert canonical_answer("3x-9=0") == ["3"]
    assert canonical_answer("x^2-5x+6=0") == ["2", "3"]
    # A bare expression has no single answer: guessing one would censor
    # correct guidance on factoring items.
    assert canonical_answer("3x-9") == []
    assert canonical_answer("explain the chain rule") == []


def test_help_request_gives_a_hint_and_remembers_the_level():
    first = _attempt("I'm stuck, give me a hint")
    second = _attempt("another hint please")
    assert first["verdict"] == second["verdict"] == "hint"
    assert first["hintLevel"] == 1 and second["hintLevel"] == 2


def test_empty_message_prompts_for_input_without_a_verdict():
    body = client.post("/api/practice/attempt", json={
        "link_id": _pub[1], "item_id": ITEM, "response": "   "}).json()
    assert body["verdict"] == "incomplete", body
    assert "step by step" in body["reply"]


def test_unparseable_attempt_asks_for_restatement_not_a_grade():
    body = _attempt("um idk it's a thing")
    assert body["verdict"] == "not_an_attempt", body
    assert body["misses"] == 0, body


def test_step_by_step_handoff_opens_a_session_on_the_item():
    # Mirrors what the "Work it step by step" button posts: the item's own
    # prompt, with no prose around it — a prefix breaks plan building.
    db = SessionLocal()
    try:
        db.query(TutorSession).delete()
        db.commit()
    finally:
        db.close()
    r = client.post("/api/tutor/message", json={
        "message": "$$3x-9=0$$",
        "pageContext": {"type": "general", "courseId": "practice-handoff",
                        "course": _course, "mode": "practice"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["replyType"] == "step", body
    assert body["totalSteps"] >= 1, body
    db = SessionLocal()
    try:
        assert db.query(TutorSession).count() == 1
    finally:
        db.close()


def test_practice_content_never_ships_the_solution_section():
    page = client.get(f"/learn/{_pub[1]}")
    assert page.status_code == 200
    assert "Answer: $3(x-3)$" not in page.text
