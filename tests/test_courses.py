"""Tests for the course catalog: options, prompt context, seeding."""
from app.courses import (
    SUPPORTED_COURSES,
    course_context,
    get_course_options,
)


def test_six_courses_present():
    assert len(SUPPORTED_COURSES) == 6
    labels = [m["label"] for m in SUPPORTED_COURSES.values()]
    for expected in ("Pre-Algebra", "Algebra 1", "Algebra 2",
                     "Trigonometry", "Pre-Calculus", "Calculus 1"):
        assert expected in labels


def test_course_options_shape():
    opts = get_course_options()
    assert len(opts) == 6
    assert all("value" in o and "label" in o for o in opts)


def test_course_context_known_and_unknown():
    ctx = course_context("Trigonometry")
    assert "Trigonometry" in ctx and "Unit circle" in ctx
    # Unknown/custom course names never crash generation prompts.
    fallback = course_context("Hyperbolic Functions")
    assert "college mathematics" in fallback
    assert course_context("") == fallback


def test_each_course_has_topics():
    for key, meta in SUPPORTED_COURSES.items():
        assert meta["topics"], f"{key} has no topics"
        assert meta["description"], f"{key} has no description"
