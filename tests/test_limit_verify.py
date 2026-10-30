"""Tests for limit verification fixes: infinity points, two-variable limits."""
from app.schemas import Question, QuestionStep
from app.verifier import _LIMIT_RE, verify_canonical


def _limit_q(prompt, answer):
    return Question(
        prompt_latex=prompt,
        steps=[QuestionStep(description="Evaluate.",
                            expected_input_latex=answer,
                            objective_tags=["limit"])],
        final_answer_latex=answer,
    )


def test_limit_regex_handles_infinity():
    m = _LIMIT_RE.search(
        "Evaluate: $$\\lim_{x \\to \\infty} \\frac{3x^2}{5x^2}$$")
    assert m is not None
    assert m.group(1) == "x"
    assert "infty" in m.group(2)


def test_limit_at_infinity_verifies():
    q = _limit_q("Evaluate: $$\\lim_{x \\to \\infty} "
                 "\\frac{3x^2+2x-1}{5x^2-4}$$", "\\frac{3}{5}")
    assert verify_canonical(q).status == "correct"


def test_two_variable_limit_verifies():
    # h -> 0 with x as a parameter (derivative-from-definition form).
    q = _limit_q("Evaluate: $$\\lim_{h \\to 0} "
                 "\\frac{(x+h)^2-x^2}{h}$$", "2x")
    assert verify_canonical(q).status == "correct"


def test_plain_limit_still_verifies():
    q = _limit_q("Evaluate: $$\\lim_{x \\to 2} "
                 "\\frac{x^2-4}{x-2}$$", "4")
    assert verify_canonical(q).status == "correct"


def test_wrong_limit_answer_rejected():
    q = _limit_q("Evaluate: $$\\lim_{x \\to 2} "
                 "\\frac{x^2-4}{x-2}$$", "5")
    assert verify_canonical(q).status == "incorrect"


def test_limit_prompt_ending_in_period_verifies():
    # Prompt prose often closes the sentence right after the math
    # ("... (3x^3 - 4x + 1)$."); the trailing "$." used to fail parsing and
    # discard an otherwise-correct generated question.
    q = _limit_q("Evaluate the limit: $\\lim_{x \\to 2} "
                 "(3x^3 - 4x + 1)$.", "17")
    assert verify_canonical(q).status == "correct"
