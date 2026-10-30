"""Verifier tests: correct / incorrect / equivalent / unsafe / unverifiable."""
from app.schemas import check_step_parseable
from app.verifier import check_equivalent, check_factored, check_numeric


def test_chained_equality_step_parses():
    # Worked steps often chain equalities; the leftover '=' used to make the
    # whole step (and the candidate question) unparseable.
    assert check_step_parseable("P(1) = 1 - 6 + 11 - 6 = 0") is None
    assert check_step_parseable("x = 2y = 8") is None


def test_trailing_sentence_period_ignored():
    assert check_equivalent("x^2.", "x^2").status == "correct"


def test_derivative_in_non_x_variable_verifies():
    # "g(t) = ...": differentiating with respect to x gave 0 and rejected a
    # correct answer, which is one reason runs delivered fewer items than asked.
    from app.schemas import Question, QuestionStep
    from app.verifier import verify_canonical

    q = Question(
        prompt_latex="Find the derivative of $g(t) = t^5 - 4t^3 + 2t - 9$.",
        final_answer_latex="g'(t) = 5t^4 - 12t^2 + 2",
        difficulty=3,
        steps=[QuestionStep(step_id="s1", description="Differentiate.",
                            expected_input_latex="5t^4 - 12t^2 + 2",
                            objective_tags=["derivative"])],
    )
    assert verify_canonical(q).status == "correct"


def test_derivative_value_at_a_point_verifies():
    # "find the marginal cost when x = 50": the answer is a number, so the
    # derivative-expression comparison alone rejected correct work.
    from app.schemas import Question, QuestionStep
    from app.verifier import verify_canonical

    def q(answer):
        return Question(
            prompt_latex=(
                "If $C(x) = 0.01x^3 - 0.6x^2 + 15x$ represents the cost of "
                "producing $x$ items, find the marginal cost when $x = 50$."),
            final_answer_latex=answer,
            difficulty=3,
            steps=[QuestionStep(step_id="s1", description="Evaluate.",
                                expected_input_latex=answer,
                                objective_tags=["derivative"])],
        )

    assert verify_canonical(q("30")).status == "correct"
    assert verify_canonical(q("45")).status == "incorrect"


def test_correct_exact():
    v = check_equivalent("(1/3)*pi*r^2*h", "(1/3)*pi*r^2*h")
    assert v.status == "correct"


def test_correct_equivalent_form():
    v = check_equivalent("pi*r^2*h/3", "(1/3)*pi*r^2*h")
    assert v.status == "correct"


def test_incorrect():
    v = check_equivalent("pi*r^2*h", "(1/3)*pi*r^2*h")
    assert v.status == "incorrect"


def test_chain_rule_step():
    v = check_equivalent("2*pi*r^2*dr/dt", "2*pi*r^2*dr/dt")
    assert v.status == "correct"


def test_missing_chain_factor_is_incorrect():
    v = check_equivalent("2*pi*r^2", "2*pi*r^2*dr/dt")
    assert v.status == "incorrect"


def test_unsafe_input_unverifiable():
    v = check_equivalent("__import__('os').system('x')", "x")
    assert v.status == "unverifiable"


def test_garbage_unverifiable():
    v = check_equivalent("??? hello", "x")
    assert v.status == "unverifiable"


def test_numeric():
    assert check_numeric("3.14159", 3.14159, tol=1e-4).status == "correct"
    assert check_numeric("3.0", 3.14159, tol=1e-4).status == "incorrect"


def test_equivalent_formats_accepted():
    expected = "(1/3)*pi*r^2*h"
    for variant in [
        "pi*r^2*h/3",
        "(pi*r^2*h)/3",
        r"\frac{1}{3}\pi r^2 h",
        r"\frac{\pi r^2 h}{3}",
        "V = (1/3)*pi*r^2*h",   # equation form
        "(1/3)πr²h",             # unicode
        "⅓πr²h",                 # vulgar fraction
        r"$\frac{1}{3}\pi r^{2}h$",
    ]:
        v = check_equivalent(variant, expected)
        assert v.status == "correct", f"{variant!r} -> {v.status}: {v.detail}"


def test_chain_rule_equivalent_formats():
    expected = "2*pi*r^2*dr/dt"
    for variant in [
        r"2\pi r^2 \frac{dr}{dt}",
        "2πr²dr/dt",
    ]:
        v = check_equivalent(variant, expected)
        assert v.status == "correct", f"{variant!r} -> {v.status}: {v.detail}"


def test_truly_wrong_still_incorrect():
    assert check_equivalent(r"\pi r^2 h", "(1/3)*pi*r^2*h").status == "incorrect"


def test_text_command_unwrapped():
    v = check_equivalent(r"x-2=0 \text{ or } x-3=0", "x-2=0")
    assert v.status in ("correct", "incorrect", "unverifiable")


def test_multi_answer_equivalent():
    v = check_equivalent("x=2, x=3", "x=3, x=2")
    assert v.status == "correct", f"{v.status}: {v.detail}"


def test_multi_answer_subset_is_incorrect():
    v = check_equivalent("x=2", "x=2, x=3")
    assert v.status == "incorrect"


def test_multi_answer_wrong_value_is_incorrect():
    v = check_equivalent("x=2, x=4", "x=2, x=3")
    assert v.status == "incorrect"


def test_double_escaped_latex_collapses():
    v = check_equivalent("3 \\\\cdot \\\\frac{sin(3x)}{3x}",
                         "3*sin(3x)/(3x)")
    assert v.status == "correct", f"{v.status}: {v.detail}"


def test_spacing_commands_ignored():
    v = check_equivalent("x\\\\,=\\\\;2", "x=2")
    assert v.status == "correct", f"{v.status}: {v.detail}"


def test_root_in_numerator_stays_a_quotient():
    """A root inside a fraction used to normalize into a product."""
    v = check_equivalent(r"\frac{\sqrt{x+4} - 2}{x}", "(sqrt(x+4)-2)/x")
    assert v.status == "correct", f"{v.status}: {v.detail}"


def test_root_in_numerator_is_not_the_product():
    v = check_equivalent(r"\frac{\sqrt{x+4} - 2}{x}", r"x(\sqrt{x+4}-2)")
    assert v.status == "incorrect"


def test_nested_fraction_converts_innermost_first():
    v = check_equivalent(r"\frac{\frac{1}{2}}{3}", "1/6")
    assert v.status == "correct", f"{v.status}: {v.detail}"


def test_factored_equation_with_zero_side():
    """Students are told to keep "= 0"; it is not a mismatch.

    The plan stores the factored form with the trailing zero, so the
    equation side had to be dropped from the student's answer as well as
    from the expected one — "(x-5)(x-2) = 0" used to grade incorrect.
    """
    exp = "(x - 5)*(x - 2) = 0"
    for answer in ("(x-5)(x-2) = 0", "(x - 5)(x - 2) = 0", "(x-2)(x-5) = 0",
                   "0 = (x-5)(x-2)", "(x-5)(x-2)"):
        v = check_factored(answer, exp)
        assert v.status == "correct", f"{answer}: {v.status} {v.detail}"


def test_factored_still_rejects_wrong_factors():
    exp = "(x - 5)*(x - 2) = 0"
    assert check_factored("(x-5)(x-3) = 0", exp).status == "incorrect"
    assert check_factored("x^2 - 7x + 10 = 0", exp).status == "incorrect"
    assert check_factored("", exp).status == "incomplete"


def test_solution_set_accepts_or_between_roots():
    """"x = 2 or x = 5" lists two roots, not one root with a stray tail.

    The parser used to split only on commas, so the second "x =" was
    swallowed and the answer came back as the single value 5. A student who
    wrote both correct roots was told they were wrong.
    """
    from app.verifier import _parse_answer_set

    items, err = _parse_answer_set("x = 2 or x = 5")
    assert err is None, err
    assert len(items) == 2, items
    assert sorted(str(i) for i in items) == ["2", "5"]


def test_solution_set_accepts_braces_and_semicolons():
    from app.verifier import _parse_answer_set

    for answer in ("{x = 2, x = 5}", "x = 2; x = 5", "x=2,x=5", "x = 2, x = 5"):
        items, err = _parse_answer_set(answer)
        assert err is None, f"{answer}: {err}"
        assert len(items) == 2, f"{answer}: {items}"


def test_solution_set_rejects_a_malformed_value_instead_of_dropping_it():
    """A second "=" is reported, never silently reduced to its right side."""
    from app.verifier import _parse_answer_set

    items, err = _parse_answer_set("x = 2 = 3")
    assert items == []
    assert err


def test_both_roots_written_with_or_are_graded_correct():
    """End to end: the answer a student actually types is graded correct."""
    from app.schemas import Question
    from app.verifier import verify_canonical

    q = Question(id=0, prompt_latex="x^2 - 7x + 10 = 0", steps=[],
                 final_answer_latex="x = 2 or x = 5")
    v = verify_canonical(q)
    assert v.status == "correct", f"{v.status}: {v.detail}"


def test_one_of_two_roots_is_still_incomplete_not_correct():
    from app.schemas import Question
    from app.verifier import verify_canonical

    q = Question(id=0, prompt_latex="x^2 - 7x + 10 = 0", steps=[],
                 final_answer_latex="x = 5")
    v = verify_canonical(q)
    assert v.status != "correct", v
