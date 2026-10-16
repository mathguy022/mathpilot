"""Tests for Plan C lesson service: escapes, derivation, verification."""
import os

os.environ["DEMO_MODE"] = "true"

from app.lessons import (  # noqa: E402
    collapse_escapes,
    derive_student_lesson,
    parse_lesson,
    verify_lesson_math,
)
from app.schemas import LessonContent, LessonSettings  # noqa: E402


def test_collapse_escapes_no_formfeed():
    # Double-escaped \\frac must collapse to \frac, never \x0c + 'rac'.
    broken = "Evaluate $\\\\frac{x^2-9}{x-3}$ and \\\\lim_{x \\\\to 3}$"
    fixed = collapse_escapes(broken)
    assert "\x0c" not in fixed
    assert "\\frac" in fixed
    assert "\\\\" not in fixed


def test_parse_lesson_collapses_escapes():
    raw = ('{"title": "Limits", "objectives": [], "prerequisites": [], '
           '"sections": [{"kind": "practice", "heading": "Evaluate", '
           '"body_markdown": "Try $\\\\\\\\lim_{x \\\\\\\\to 3} '
           '\\\\\\\\frac{x}{y}$", "math_latex": [], "examples": []}]}')
    les = parse_lesson(raw)
    assert "\\frac" in les.sections[0].body_markdown
    assert "\x0c" not in les.sections[0].body_markdown


def test_derive_student_lesson_strips_solutions():
    les = LessonContent(
        title="Limits", course="Calculus", topic="Limits",
        objectives=["Evaluate limits"],
        sections=[
            {"kind": "explanation", "heading": "Idea",
             "body_markdown": "A limit is $\\\\lim$.", "math_latex": []},
            {"kind": "practice", "heading": "Try",
             "body_markdown": "Evaluate $x$.",
             "math_latex": ["\\\\lim_{x \\\\to 1} x"], "examples": []},
            {"kind": "solutions", "heading": "Keys",
             "body_markdown": "Answer: $1$.", "math_latex": [],
             "examples": []},
        ],
    )
    student = derive_student_lesson(les, LessonSettings())
    kinds = [s["kind"] for s in student["sections"]]
    assert "solutions" not in kinds
    assert len(student["guided_items"]) == 1
    blob = str(student)
    assert "Answer: $1$" not in blob


def test_verify_lesson_math_honest_counts():
    les = LessonContent(
        title="T", course="C", topic="T",
        sections=[{
            "kind": "worked_example", "heading": "Ex",
            "body_markdown": "", "math_latex": [],
            "examples": [
                {"problem_latex": "Simplify",
                 "steps": [{"expression": "x+1",
                            "why": "given"}],
                 "final_answer_latex": "x+1"},
                {"problem_latex": "Broken",
                 "steps": [{"expression": "@@@",
                            "why": "bad"}],
                 "final_answer_latex": "@@@"},
            ],
        }],
    )
    counts = verify_lesson_math(les)
    assert counts["verified"] == 1
    assert counts["verified"] + counts["needs_review"] + counts["unable"] == 2
