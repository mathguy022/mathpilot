"""Central course catalog: single source of truth for supported courses.

The instructor dashboard dropdown, the database seed, and the AI prompt
context all read from SUPPORTED_COURSES. Course labels double as the
human-readable names stored on Course rows.
"""
from __future__ import annotations

SUPPORTED_COURSES: dict[str, dict] = {
    "pre-algebra": {
        "label": "Pre-Algebra",
        "description": "Foundations: integers, fractions, decimals, basic equations.",
        "topics": ["Order of Operations", "Fractions & Decimals",
                   "Basic Linear Equations"],
    },
    "algebra-1": {
        "label": "Algebra 1",
        "description": "Linear systems, inequalities, polynomials, factoring.",
        "topics": ["Linear Equations", "Systems of Equations",
                   "Factoring Polynomials", "Quadratics"],
    },
    "algebra-2": {
        "label": "Algebra 2",
        "description": "Complex numbers, logarithms, advanced functions, conics.",
        "topics": ["Logarithms & Exponentials", "Complex Numbers",
                   "Conic Sections", "Rational Functions"],
    },
    "trigonometry": {
        "label": "Trigonometry",
        "description": "Unit circle, identities, laws of sines/cosines.",
        "topics": ["Unit Circle Basics", "Trig Identities",
                   "Law of Sines/Cosines", "Inverse Trig"],
    },
    "pre-calculus": {
        "label": "Pre-Calculus",
        "description": "Functions, limits intro, vectors, polar coordinates.",
        "topics": ["Function Analysis", "Vectors & Matrices",
                   "Polar Coordinates", "Intro to Limits"],
    },
    "calculus-1": {
        "label": "Calculus 1",
        "description": "Derivatives, applications, integrals, FTC.",
        "topics": ["Limits & Continuity", "Derivatives",
                   "Applications of Derivatives", "Integrals"],
    },
}


def get_course_options() -> list[dict[str, str]]:
    """Dropdown options: [{"value": key, "label": label}]."""
    return [{"value": key, "label": val["label"]}
            for key, val in SUPPORTED_COURSES.items()]


def get_course_topics() -> dict[str, list[str]]:
    """Label -> standard topics, for the dashboard's "Try: …" suggestions.

    Rendered into the page rather than hand-copied into the template: the copy
    is what let a retired course label keep appearing in the UI.
    """
    return {val["label"]: list(val["topics"])
            for val in SUPPORTED_COURSES.values()}


def course_context(course_label: str) -> str:
    """Curriculum hint line for generation prompts.

    Matches on label (case-insensitive); unknown labels yield a neutral
    line so generation never crashes on a custom course name.
    """
    needle = (course_label or "").strip().lower()
    for meta in SUPPORTED_COURSES.values():
        if meta["label"].lower() == needle:
            topics = ", ".join(meta["topics"])
            return (f"You are writing for {meta['label']}: "
                    f"{meta['description']} Standard topics include: "
                    f"{topics}. Keep difficulty appropriate to this level.")
    return ("You are writing for a college mathematics course. "
            "Keep difficulty appropriate to the stated course and topic.")
