"""Tutoring orchestration. Enforces hint policy + reveal policy in code."""
from __future__ import annotations

from . import config, nebius
from .verifier import check_equivalent

HINT_QUESTIONS = [
    "What information does the problem give you, and what are you asked to find?",
    "Which mathematical relationship connects those quantities?",
    "What happens if you apply that relationship to both sides here?",
    "Can you explain why that step is valid before moving on?",
]


def demo_reply(step: str, verdict: str, hint_level: int, fails: int) -> str:
    """Labeled fallback used when no API key is configured."""
    if verdict == "correct":
        return (f"**Demo mode** — Nice work! That step checks out. "
                f"What is the next step after `{step}`?")
    if hint_level >= 1:
        q = HINT_QUESTIONS[min(hint_level - 1, len(HINT_QUESTIONS) - 1)]
        return (f"**Demo mode** — Not quite yet. Hint (level {hint_level}): {q} "
                f"(attempt {fails}). Add your Nebius key for live Socratic tutoring.")
    return ("**Demo mode** — Try breaking it into a smaller step. "
            "Add your Nebius key for live Socratic tutoring.")


def orchestrate(problem: str, step: str, expected: str, student: str,
                step_index: int, total_steps: int, hint_level: int,
                fails: int, allow_reveal: bool, max_attempts: int,
                history: list[dict]) -> dict:
    verdict = check_equivalent(student, expected)
    ai_judge = "n/a"
    if config.ai_configured():
        try:
            ai_judge = nebius.judge_step(step, expected, student)
        except Exception as exc:
            ai_judge = f"judge unavailable: {exc}"

    forced_reveal = (
        allow_reveal and verdict.status != "correct" and fails + 1 >= max_attempts
    )
    if forced_reveal:
        reply = (f"You've worked hard on this step. The expected form is "
                 f"`{expected}`. Can you explain in your own words why that works?")
        return {"verdict": "incorrect", "detail": verdict.detail,
                "reply": reply, "revealed": True, "ai_judge": ai_judge}

    if config.ai_configured():
        try:
            reply = nebius.socratic_reply(
                {"problem": problem, "step": step, "expected": expected,
                 "student": student, "verdict": verdict.status,
                 "verdict_detail": verdict.detail, "ai_judge": ai_judge,
                 "hint_level": hint_level, "step_index": step_index,
                 "total_steps": total_steps, "allow_reveal": allow_reveal},
                history,
            )
        except Exception as exc:
            reply = demo_reply(step, verdict.status, hint_level, fails + 1) + \
                f" (live tutor unavailable: {exc})"
    else:
        reply = demo_reply(step, verdict.status, hint_level, fails + 1)

    return {"verdict": verdict.status, "detail": verdict.detail,
            "reply": reply, "revealed": False, "ai_judge": ai_judge}
