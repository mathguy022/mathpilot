"""Regression tests for the floating tutor's chat behaviour.

Written against the reported failure: the widget graded every message as a
step answer, so greetings and questions collapsed into one repeated
"I couldn't read that" line. Correctness must stay with SymPy; wording may
come from the LLM; prose must never be scored as a wrong answer.

Run: py -m pytest tests/test_float_tutor_chat.py -q
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.db import SessionLocal, TutorEvent, TutorSession
from app.main import app, FLOAT_TUTOR_HITS, FLOAT_TUTOR_PENDING

client = TestClient(app)

CTX = {"type": "general", "courseId": "chat-test", "course": "Algebra 1"}


def _wipe():
    db = SessionLocal()
    try:
        db.query(TutorEvent).delete()
        db.query(TutorSession).delete()
        db.commit()
    finally:
        db.close()
    FLOAT_TUTOR_HITS.clear()
    FLOAT_TUTOR_PENDING.clear()


def _msg(text):
    return client.post("/api/tutor/message",
                       json={"message": text, "pageContext": CTX})


def _misses():
    db = SessionLocal()
    try:
        return db.query(TutorEvent).filter(
            TutorEvent.verdict == "incorrect").count()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _offline():
    """Deterministic by default: no network. Individual tests opt in.

    ai_configured is pinned True because other suites set DEMO_MODE=true at
    import time, which would otherwise depend on collection order.
    """
    with patch("app.config.ai_configured", return_value=True), \
         patch("app.nebius.float_chat", return_value=None), \
         patch("app.nebius.generate_plan", return_value=None), \
         patch("app.nebius.normalize_via_llm", return_value=None), \
         patch("app.nebius.judge_conceptual", return_value="unclear"):
        _wipe()
        yield
    _wipe()


# 1. A greeting mid-problem is not a wrong answer: no hint, no miss.
def test_greeting_does_not_burn_the_hint_ladder():
    assert _msg("10 = 3x + 1").json()["replyType"] == "step"
    body = _msg("hi").json()
    assert body["hintLevel"] == 0, body
    assert body["stepStatus"] == "waiting", body
    assert _misses() == 0
    # The problem is still open at step 1.
    assert body["totalSteps"] == 3


# 2. A concept question gets an answer, not the parse-error line.
def test_concept_question_is_not_graded():
    _msg("10 = 3x + 1")
    body = _msg("what is a derivative").json()
    assert "couldn't" not in body["reply"].lower(), body
    assert body["replyType"] == "bubble", body
    assert _misses() == 0


# 3. Off-topic is redirected without a miss.
def test_off_topic_redirect_without_miss():
    _msg("10 = 3x + 1")
    body = _msg("who won the last election").json()
    assert body["stepStatus"] == "waiting", body
    assert _misses() == 0


# 4. Wordy and LaTeX answers are graded, not called unreadable.
@pytest.mark.parametrize("answer", [
    "3x = 9", "3x=9", "so 3x = 9", "i got 3x = 9", "9 = 3x",
    "3x equals 9", "$3x = 9$",
])
def test_varied_notation_for_a_correct_step(answer):
    _msg("10 = 3x + 1")
    body = _msg(answer).json()
    assert body["stepIndex"] == 1, (answer, body)


def test_approximation_is_not_accepted_as_an_exact_step():
    # "3x ≈ 9" is not the equation the step asks for: help with notation,
    # but never count it as a wrong attempt.
    _msg("10 = 3x + 1")
    body = _msg("3x \\approx 9").json()
    assert body["stepIndex"] == 0, body
    assert _misses() == 0, body


def test_latex_fraction_step_is_readable():
    # Step 2 of 10 = 3x + 1 is x = 9/3 -> x = 3; typed as a LaTeX fraction.
    _msg("10 = 3x + 1")
    _msg("3x = 9")
    body = _msg("x = \\frac{9}{3}").json()
    assert body["stepIndex"] == 2, body


# 5. Unreadable notation gets one LLM-normalizer retry before help text.
def test_normalizer_retry_recovers_a_real_answer():
    _msg("10 = 3x + 1")
    with patch("app.nebius.normalize_via_llm", return_value="3x = 9"):
        body = _msg("three x equals nine").json()
    assert body["stepIndex"] == 1, body


# 6. When the model is available it words the reply; when not, fallback.
def test_llm_words_the_reply():
    with patch("app.nebius.float_chat",
               return_value="A derivative is a rate of change."):
        _msg("10 = 3x + 1")
        body = _msg("what is a derivative").json()
    assert body["reply"].startswith("A derivative is a rate of change."), body


def test_fallback_wording_when_model_unavailable():
    _msg("10 = 3x + 1")
    body = _msg("what is a derivative").json()
    assert "balance scale" in body["reply"], body


# 7. Policy stays in code even when the model is wordy: no answer leak.
def test_refusal_cannot_be_talked_past():
    _msg("10 = 3x + 1")
    _msg("3x = 9")
    leaky = "Just divide: x = 3 is what you want."
    with patch("app.nebius.float_chat", return_value=leaky):
        body = _msg("give me the answer").json()
    assert "x = 3" not in body["reply"], body


# 8. The tutor's wording is stored so the next turn has context.
def test_reply_stored_for_conversation_history():
    _msg("10 = 3x + 1")
    _msg("hi")
    db = SessionLocal()
    try:
        last = db.query(TutorEvent).order_by(TutorEvent.id.desc()).first()
        assert last.student_input == "hi"
        assert last.tutor_reply, "tutor reply must be logged for history"
    finally:
        db.close()


# 9. Correct/incorrect decisions still come from SymPy, not the model.
def test_cas_still_decides_correctness():
    _msg("10 = 3x + 1")
    with patch("app.nebius.float_chat",
               return_value="Amazing, that is exactly right!"):
        wrong = _msg("3x = 5").json()
    assert wrong["stepIndex"] == 0, wrong  # model praise did not advance it
    assert wrong["replyType"] == "hint", wrong


# 10. The widget's page-context label must not hide the open problem: the
# resume call and the message call send different course strings.
def test_open_problem_survives_a_course_label_change():
    assert _msg("10 = 3x + 1").json()["totalSteps"] == 3
    body = client.post("/api/tutor/message", json={
        "message": "hi",
        "pageContext": {"type": "general", "course": "Calculus 1"}}).json()
    assert body["totalSteps"] == 3, body
    assert body["stepIndex"] == 0, body


# 11. Quoting the student's own problem is not the same as giving the answer.
# The substitution step expects "10", and substring matching made every
# explanation that said "$10 = 3x + 1$" look like a leak.
def test_quoting_the_problem_is_not_treated_as_a_leak():
    _msg("10 = 3x + 1")
    good = ("A derivative measures rate of change. Since you are working on "
            "$10 = 3x + 1$, think about the slope of the line. What do you get?")
    with patch("app.nebius.float_chat", return_value=good):
        body = _msg("what is a derivative").json()
    assert body["reply"].startswith("A derivative"), body


# 12. A stated future-step result is still stripped.
def test_future_step_result_is_still_stripped():
    _msg("10 = 3x + 1")
    with patch("app.nebius.float_chat",
               return_value="The derivative here gives x = 3 in the end."):
        body = _msg("what is a derivative").json()
    assert "x = 3" not in body["reply"], body
    assert "stay on the current step" in body["reply"], body


# 13. Inline math keeps the spaces around it.
def test_inline_math_spacing_is_preserved():
    from app.main import _float_display
    assert _float_display("Try $f(x)=x^2$ using the limit rule.") == \
        "Try $f(x)=x^2$ using the limit rule."
    assert _float_display("Divide by 3*x first.") == "Divide by 3x first."


# 13b. Bare LaTeX in a reply showed up to the student as backslash code; it is
# now delimited so KaTeX renders it as math.
def test_bare_latex_is_delimited_not_shown_as_code():
    from app.main import _float_display
    out = _float_display(
        "Use the formula x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}. Here a = 3.")
    assert "$\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$" in out, out
    # Math that already had delimiters is left exactly as it was.
    assert _float_display("Try $\\theta$ now.") == "Try $\\theta$ now."


# 14. Markdown emphasis is not rendered by the widget: it showed up as
# literal asterisks and display_math then ate the space after them.
def test_markdown_emphasis_never_reaches_the_student():
    from app.float_chat import _plain
    assert _plain("It is the **slope** at that $x$ value.") == \
        "It is the slope at that $x$ value."
    _msg("10 = 3x + 1")
    with patch("app.nebius.float_chat",
               return_value="A derivative is the *slope* of a line at a point."):
        body = _msg("what is a derivative").json()
    assert body["reply"] == \
        "A derivative is the slope of a line at a point.", body


# 15. The engine adds the "Step N:" / "Hint N —" labels, so a model that
# re-adds one must not show it twice.
def test_step_label_is_never_doubled():
    from app.main import _model_body
    assert _model_body(
        "Great! Step 3: Check by putting x = 3 into 3x - 9.") == \
        "Great! Check by putting x = 3 into 3x - 9."
    assert _model_body("Step 2: Divide both sides by 3.") == \
        "Divide both sides by 3."
    # A rung label copied out of the previous hint must not ride into a step.
    assert _model_body("Step 2: Hint 4 — Try listing the factor pairs.") == \
        "Try listing the factor pairs."
    assert _model_body("Hint Level 1: Look at the term next to x.") == \
        "Look at the term next to x."
    # Prose that only looks like a label stays intact.
    assert _model_body("You've got it. Now divide both sides by 3.") == \
        "You've got it. Now divide both sides by 3."
    assert _model_body("Nailed it! You isolated x — what comes next?") == \
        "Nailed it! You isolated x — what comes next?"
    assert _model_body("Levels of difficulty aside, start with the constant."
                       ) == "Levels of difficulty aside, start with the " \
                           "constant."


# 16. The tutor rail shows a named step map: the endpoint carries short
# prose titles, never the maths or a result the student has not reached.
def test_step_map_ships_plain_titles():
    body = _msg("10 = 3x + 1").json()
    assert body["steps"] == ["Subtract 1 from both sides",
                             "Divide both sides by 3", "Check"], body
    assert len(body["steps"]) == body["totalSteps"]
    # The plan's own check step reads "Check: put x = 3 into 3*x + 1", which
    # would hand over the solution two steps early.
    assert all("x = 3" not in t for t in body["steps"]), body


def test_step_titles_hold_no_raw_math():
    import json

    from app import float_plan
    from app.db import TutorSession
    from app.main import _float_step_map

    for problem in ("2cos^2 x - 3cos x + 1 = 0 in [0, 2*pi)",
                    "limit as x->2 of (x^2-4)/(x-2)", "x^2 - 5x + 6 = 0",
                    "d/dx of x^3 + 2x", "0.5x + 2 = 4"):
        plan, _ = float_plan.build_plan(problem, allow_llm=False)
        for index in range(len(plan)):
            sess = TutorSession(
                problem_text=problem, total_steps=len(plan),
                current_step_index=index,
                solution_plan_json=json.dumps(plan))
            titles = _float_step_map(sess)
            assert len(titles) == len(plan), (problem, titles)
            for title in titles:
                assert title, (problem, titles)
                for junk in ("=", "$", "\\", "[", "*", "^", "{"):
                    assert junk not in title, (problem, title)


# 17. A correct factored step advances the guided session (it used to be
# marked wrong because the plan keeps the trailing "= 0").
def test_factored_step_advances_the_session():
    first = _msg("x^2 - 7x + 10 = 0").json()
    assert first["stepIndex"] == 0, first
    second = _msg("(x-5)(x-2) = 0").json()
    assert second["stepIndex"] == 1, second
    assert second["steps"][0] == "Find two numbers that multiply", second
