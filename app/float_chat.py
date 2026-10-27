"""Course-scoped conversational replies for the floating tutor.

The CAS (SymPy) decides whether a step is correct; this module only words
replies. Every helper returns None when the LLM is unavailable so callers
fall back to their deterministic wording — the widget never dead-ends.
"""
from __future__ import annotations

import re

from . import config, courses, nebius

# The tutor answers mathematics from this sequence only.
TUTOR_COURSES = ("pre-algebra", "algebra-1", "algebra-2", "trigonometry",
                 "pre-calculus", "calculus-1")

_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}


def _normalize_label(text: str) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    t = re.sub(r"\bcalculus\s+([ivx]+)\b",
               lambda m: "calculus " + _ROMAN.get(m.group(1), m.group(1)), t)
    t = re.sub(r"\balgebra\s+([ivx]+)\b",
               lambda m: "algebra " + _ROMAN.get(m.group(1), m.group(1)), t)
    return re.sub(r"\s+", " ", t)


def resolve_course(*refs: str) -> dict:
    """Map a course label/id from the client onto the catalog entry.

    Advisory only: an unknown label still yields the full supported
    sequence, so the scope gate never widens because of bad context.
    """
    needles = {_normalize_label(r) for r in refs if r}
    needles |= {str(r or "").strip().lower() for r in refs if r}
    for key in TUTOR_COURSES:
        meta = courses.SUPPORTED_COURSES[key]
        hay = {_normalize_label(meta["label"]), key,
               _normalize_label(meta["description"])}
        if needles & hay:
            return {"key": key, **meta}
    return {}


def scope_line(course: dict) -> str:
    if course:
        return (
            f"The student's course is {course['label']} "
            f"({course['description']} Typical topics: "
            f"{', '.join(course['topics'])})."
        )
    labels = ", ".join(courses.SUPPORTED_COURSES[k]["label"]
                       for k in TUTOR_COURSES)
    return f"The student has not told us the course. Cover: {labels}."


def system_prompt(*, course: dict, intent: str, situation: str,
                  graded: bool) -> str:
    labels = ", ".join(courses.SUPPORTED_COURSES[k]["label"]
                       for k in TUTOR_COURSES)
    rules = [
        "You are MathPilot, a friendly, encouraging mathematics tutor chatting "
        "with one student in a browser widget.",
        "Output ONLY the message the student reads — no thinking, no labels, "
        "no markdown headings, no * or ** used for emphasis.",
        nebius.MATH_LATEX_RULE,
        f"Stay under 80 words. Ask at most ONE question, at the end.",
        scope_line(course),
        f"You help with mathematics from this course sequence ONLY: {labels}. "
        "If the message is not mathematics, or belongs beyond these courses "
        "(statistics, chemistry, essay writing, current events), say so in "
        "one short sentence and steer back to a topic you do cover. Never "
        "invent course materials, due dates, or grades.",
        "Never state the final answer to the problem the student is working, "
        "and never write out the remaining steps. Guide with the next move "
        "only.",
    ]
    if graded:
        rules.append("This is GRADED work: hints only, no answer, no worked "
                     "substeps beyond the current instruction.")
    if intent == "concept":
        rules.append("This is a conceptual question. Explain the idea "
                     "briefly and correctly, then invite the student to try "
                     "one step themselves.")
    if intent == "off_topic":
        rules.append("Redirect warmly to course math in one or two sentences; "
                     "do not answer the off-topic request.")
    return "Rules:\n" + "\n".join(f"- {r}" for r in rules) + \
        f"\n\nSituation: {situation}"


def _plain(text: str | None) -> str | None:
    """Drop markdown emphasis the widget renders as literal asterisks.

    Only outside $...$ regions, so LaTeX like $x^2$ is untouched.
    """
    if not text:
        return text
    parts = text.split("$")
    for i in range(0, len(parts), 2):
        s = parts[i]
        s = re.sub(r"^\s*#+\s*", "", s, flags=re.M)      # headings
        s = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"\1", s, flags=re.S)
        s = re.sub(r"(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)", r"\1", s)
        parts[i] = s
    return "$".join(parts)


def reply(*, user_text: str, history: list[dict], course_ref: str = "",
          intent: str = "chat", situation: str = "",
          graded: bool = False) -> str | None:
    """One conversational tutor message, or None if the LLM is unusable."""
    if not config.ai_configured() or not user_text.strip():
        return None
    return _plain(nebius.float_chat(
        system_prompt(course=resolve_course(course_ref), intent=intent,
                      situation=situation or "General help conversation.",
                      graded=graded),
        history, user_text))
