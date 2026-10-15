"""Verifier tests: correct / incorrect / equivalent / unsafe / unverifiable."""
from app.verifier import check_equivalent, check_numeric


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
