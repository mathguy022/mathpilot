"""Authoring-layer schemas: Question/QuestionStep/LessonPlan + Activity adapter.

Phase 1: these models validate and convert to/from the existing
Activity/expected_steps_json storage. No DB migration. Policy columns
(allow_solution_reveal, max_attempts_before_reveal, accepted_methods)
are preserved on every conversion.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QuestionStep(BaseModel):
    step_id: str = ""
    description: str
    expected_input_latex: str
    accepted_equivalents: list[str] = []
    hints: list[str] = []
    objective_tags: list[str] = []


class TutoringPolicy(BaseModel):
    allow_solution_reveal: bool = True
    max_attempts_before_reveal: int = 3
    accepted_methods: str = ""


class Question(BaseModel):
    id: int = 0
    prompt_latex: str
    difficulty: int = Field(default=3, ge=1, le=5)
    steps: list[QuestionStep]
    final_answer_latex: str
    policy: TutoringPolicy = TutoringPolicy()
    source: str = "instructor"  # instructor | ai-generated
    is_verified: bool = False


class LessonPlan(BaseModel):
    course_id: str = ""
    section_title: str = ""
    notes_markdown: str = ""
    worked_example_latex: str = ""
    questions: list[Question] = []


def step_to_legacy(step: QuestionStep) -> dict[str, Any]:
    """Convert one QuestionStep to an expected_steps_json entry."""
    return {
        "prompt": step.description,
        "expected": step.expected_input_latex,
        "hint": list(step.hints),
        "accepted_equivalents": list(step.accepted_equivalents),
        "objective_tags": list(step.objective_tags),
    }


def step_from_legacy(entry: dict[str, Any], index: int) -> QuestionStep:
    """Convert one expected_steps_json entry to a QuestionStep."""
    return QuestionStep(
        step_id=str(entry.get("step_id", f"step-{index + 1}")),
        description=entry.get("prompt", ""),
        expected_input_latex=entry.get("expected", ""),
        accepted_equivalents=list(entry.get("accepted_equivalents", [])),
        hints=list(entry.get("hint", [])),
        objective_tags=list(entry.get("objective_tags", [])),
    )


def question_to_activity_fields(question: Question) -> dict[str, Any]:
    """Map a Question onto Activity columns (title/problem/steps/policy).

    Preserves tutoring policy; never drops it. Caller writes the result
    onto an Activity row.
    """
    import json

    return {
        "problem_statement": question.prompt_latex,
        "expected_steps_json": json.dumps(
            [step_to_legacy(s) for s in question.steps]
        ),
        "accepted_methods": question.policy.accepted_methods,
        "allow_solution_reveal": question.policy.allow_solution_reveal,
        "max_attempts_before_reveal": question.policy.max_attempts_before_reveal,
    }


def question_from_activity(activity: Any) -> Question:
    """Build a Question view from an Activity row (lossless for steps+policy)."""
    import json

    try:
        entries = json.loads(activity.expected_steps_json or "[]")
    except Exception:
        entries = []
    steps = [step_from_legacy(e, i) for i, e in enumerate(entries)]
    return Question(
        id=activity.id or 0,
        prompt_latex=activity.problem_statement or "",
        steps=steps,
        final_answer_latex=steps[-1].expected_input_latex if steps else "",
        policy=TutoringPolicy(
            allow_solution_reveal=bool(activity.allow_solution_reveal),
            max_attempts_before_reveal=activity.max_attempts_before_reveal or 3,
            accepted_methods=activity.accepted_methods or "",
        ),
    )


def split_answers(text: str) -> list[str]:
    """Split a multi-answer string on top-level commas: 'x=2, x=3' -> [...].

    Commas nested inside braces/parens (e.g. \\frac{a,b}{c}) do not split.
    """
    parts, depth, current = [], 0, ""
    for ch in text:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    parts.append(current.strip())
    return [p for p in parts if p]


def _parseable(text: str) -> str | None:
    """Return None if text parses under the verifier, else the error."""
    from .verifier import _safe_parse

    try:
        _safe_parse(text)
    except Exception as exc:
        return str(exc)
    return None


def check_step_parseable(step_text: str) -> str | None:
    """A step passes if it parses whole OR every comma-separated answer parses."""
    err = _parseable(step_text)
    if err is None:
        return None
    alternatives = split_answers(step_text)
    if len(alternatives) > 1 and all(
        _parseable(a) is None for a in alternatives
    ):
        return None
    return err


def verify_question(question: Question) -> tuple[bool, list[str]]:
    """Deterministic verification gate for is_verified.

    Each step's expected_input_latex and accepted_equivalents must parse
    under the verifier allowlist (multi-answer steps like 'x=2, x=3' pass
    if every answer parses); final_answer_latex must be algebraically
    equivalent to the last step's expectation. Returns (ok, reasons).
    """
    from .verifier import check_equivalent

    reasons: list[str] = []
    if not question.steps:
        return False, ["no steps to verify"]
    for i, step in enumerate(question.steps):
        err = check_step_parseable(step.expected_input_latex)
        if err is not None:
            reasons.append(f"step {i + 1} expected input unparseable: {err}")
        for j, alt in enumerate(step.accepted_equivalents):
            err = check_step_parseable(alt)
            if err is not None:
                reasons.append(
                    f"step {i + 1} equivalent {j + 1} unparseable: {err}"
                )
    if question.final_answer_latex and question.steps:
        verdict = check_equivalent(
            question.final_answer_latex,
            question.steps[-1].expected_input_latex,
        )
        if verdict.status != "correct":
            reasons.append(
                f"final answer not equivalent to last step: {verdict.detail}"
            )
    return (not reasons), reasons


def verify_question_canonical(question: Question) -> tuple[bool, list[str]]:
    """Canonical-solver gate: SymPy independently solves the prompt.

    Step expectations only need to PARSE (they are partial work, not the
    full solution). The final answer is compared against the canonical
    solution via verifier.verify_canonical. Returns (ok, reasons).
    """
    from .verifier import verify_canonical

    reasons: list[str] = []
    if not question.steps:
        return False, ["no steps to verify"]
    for i, step in enumerate(question.steps):
        err = check_step_parseable(step.expected_input_latex)
        if err is not None:
            reasons.append(f"step {i + 1} expected input unparseable: {err}")
        for j, alt in enumerate(step.accepted_equivalents):
            err = check_step_parseable(alt)
            if err is not None:
                reasons.append(
                    f"step {i + 1} equivalent {j + 1} unparseable: {err}"
                )
    if reasons:
        return False, reasons
    verdict = verify_canonical(question)
    if verdict.status != "correct":
        reasons.append(f"canonical check: {verdict.detail}")
        return False, reasons
    return True, reasons
