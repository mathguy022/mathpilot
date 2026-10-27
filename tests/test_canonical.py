"""Tests for the canonical solver (Phase 3): equations, systems, calculus."""
from app.schemas import Question, QuestionStep, verify_question_canonical
from app.verifier import detect_mode, verify_canonical


def _q(prompt, answer, tags=None, steps=None):
    steps = steps if steps is not None else [
        QuestionStep(description="Solve.", expected_input_latex=answer,
                     objective_tags=list(tags or [])),
    ]
    return Question(prompt_latex=prompt, steps=steps,
                    final_answer_latex=answer)


def test_detect_mode_explicit_tags():
    q = _q("2x+3=7", "x=2", tags=["linear equation"])
    assert detect_mode(q) == "equation"
    q = _q("x+y=3", "x=1, y=2", tags=["linear system"])
    assert detect_mode(q) == "system"
    q = _q("d/dx x^2", "2x", tags=["derivative"])
    assert detect_mode(q) == "deriv"


def test_detect_mode_defaults_to_equiv():
    q = _q("x+1=2", "x=1")
    assert detect_mode(q) == "equiv"
    v = verify_canonical(q)
    assert v.status == "correct"


def test_single_linear_equation():
    q = _q("2x+3=7", "x=2", tags=["linear equation"])
    assert verify_canonical(q).status == "correct"


def test_equation_with_prose_lead_in():
    q = _q("Solve x+y=3, x-y=1", "x=2, y=1", tags=["linear system"])
    assert verify_canonical(q).status == "correct"


def test_cases_environment_prompt():
    prompt = ("Solve the following system of linear equations:\n"
              "$$\n\\begin{cases}\n2x + y = 7 \\\\\n x - y = 2\n"
              "\\end{cases}\n$$")
    q = _q(prompt, "x=3, y=1", tags=["linear system"])
    assert verify_canonical(q).status == "correct"


def test_single_equation_wrong():
    q = _q("2x+3=7", "x=3", tags=["linear equation"])
    assert verify_canonical(q).status == "incorrect"


def test_quadratic_two_solutions():
    q = _q("x^2-5x+6=0", "x=2, x=3", tags=["quadratic"])
    assert verify_canonical(q).status == "correct"


def test_quadratic_subset_is_incorrect():
    q = _q("x^2-5x+6=0", "x=2", tags=["quadratic"])
    assert verify_canonical(q).status == "incorrect"


def test_no_solution():
    q = _q("x=x+1", "x=0", tags=["linear equation"])
    v = verify_canonical(q)
    assert v.status == "incorrect"


def test_linear_system_unique():
    q = _q("x+y=3, x-y=1", "x=2, y=1", tags=["linear system"])
    assert verify_canonical(q).status == "correct"


def test_linear_system_wrong():
    q = _q("x+y=3, x-y=1", "x=1, y=1", tags=["linear system"])
    assert verify_canonical(q).status == "incorrect"


def test_extraneous_root_rejected():
    # 1/(x-1) = 2/(x^2-1): x=-1 makes a denominator zero.
    q = _q("1/(x-1) = 2/(x^2-1)", "x=0, x=-1",
           tags=["extraneous roots"])
    v = verify_canonical(q)
    assert v.status in ("incorrect", "unverifiable")


def test_derivative():
    q = _q("f(x) = x^3+2x", "3x^2+2", tags=["derivative"])
    assert verify_canonical(q).status == "correct"


def test_derivative_wrong():
    q = _q("f(x) = x^3+2x", "3x^2", tags=["derivative"])
    assert verify_canonical(q).status == "incorrect"


def test_integral_ignores_constant():
    q = _q("f(x) = 2x", "x^2+5", tags=["integral"])
    assert verify_canonical(q).status == "correct"


def test_integral_wrong():
    q = _q("f(x) = 2x", "x^3", tags=["integral"])
    assert verify_canonical(q).status == "incorrect"


def test_limit_supported_grammar():
    q = _q(r"\lim_{x \to 2} x^2+1", "5", tags=["limit"])
    assert verify_canonical(q).status == "correct"


def test_limit_unsupported_grammar_unverifiable():
    q = _q("what is the limit of x^2 near 2?", "4", tags=["limit"])
    assert verify_canonical(q).status == "unverifiable"


def test_simplify_equivalent():
    q = _q("(x+1)^2-x^2", "2x+1", tags=["simplify"])
    assert verify_canonical(q).status == "correct"


def test_simplify_wrong():
    q = _q("(x+1)^2-x^2", "2x+2", tags=["simplify"])
    assert verify_canonical(q).status == "incorrect"


def test_canonical_gate_accepts_good_question():
    q = _q("x^2-5x+6=0", "x=2, x=3", tags=["quadratic"])
    ok, reasons = verify_question_canonical(q)
    assert ok, reasons


def test_canonical_gate_rejects_bad_answer():
    q = _q("x^2-5x+6=0", "x=1, x=1", tags=["quadratic"])
    ok, reasons = verify_question_canonical(q)
    assert not ok
    assert reasons


def test_display_math_keeps_zero():
    """display_math took its argument through `text or ""`, so 0 vanished."""
    import sympy as sp

    from app.verifier import canonical_answer, display_math

    assert display_math(sp.Integer(0)) == "0"
    assert canonical_answer(
        r"\lim_{x \to 0} \frac{\sqrt{x+4} - 2}{x}") == ["1/4"]
