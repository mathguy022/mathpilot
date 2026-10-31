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


# --- Plan C: structured lesson + assignment contracts ---

SECTION_KINDS = (
    "objectives", "prerequisites", "introduction", "definitions",
    "explanation", "worked_example", "visualization", "misconceptions",
    "practice", "solutions", "summary", "comprehension_check",
)


class LessonExample(BaseModel):
    """One worked example: problem + ordered steps with why-valid notes."""

    problem_latex: str = ""
    steps: list[dict[str, str]] = []  # {expression, why}
    final_answer_latex: str = ""
    verification: str = "needs-review"  # verified|needs-review|unable-to-verify


class LessonSection(BaseModel):
    kind: str = "explanation"
    heading: str = ""
    body_markdown: str = ""
    math_latex: list[str] = []
    examples: list[LessonExample] = []


class LessonContent(BaseModel):
    """Full instructor lesson: structured, renderable, editable sections."""

    title: str = ""
    course: str = ""
    topic: str = ""
    objectives: list[str] = []
    prerequisites: list[str] = []
    sections: list[LessonSection] = []
    source_notes: str = ""  # derived-from-source vs ai-supplemental marker
    verification_summary: dict[str, int] = {}


class LessonSettings(BaseModel):
    hints_enabled: bool = True
    reveal_policy: str = "after-attempts"  # immediate|after-attempts|never
    allow_retry: bool = True
    require_comprehension: bool = False
    record_completion: bool = True
    max_attempts_before_reveal: int = 3


class AssignmentItem(BaseModel):
    item_id: str = ""
    prompt_latex: str = ""
    steps: list[QuestionStep] = []
    hints: list[str] = []
    objective_tags: list[str] = []
    difficulty: int = Field(default=3, ge=1, le=5)
    verification: str = "needs-review"  # verified|needs-review|unable-to-verify


class AssignmentSolution(BaseModel):
    final_answer_latex: str = ""
    solution_steps: list[str] = []
    explanation: str = ""


class AssignmentSettings(BaseModel):
    title: str = ""
    due_date: str = ""
    attempts_allowed: int = 1
    grading_policy: str = "Completion"
    hints_enabled: bool = True
    solution_release: str = "after-due"  # after-due|after-attempts|never


def lesson_assignment_context(content: "LessonContent") -> str:
    """Distil a lesson into generator notes so questions match what was taught.

    Passing only the objectives let the model invent off-topic problems. This
    digest carries the section headings, the worked-example problems (with
    their answers) and the practice problems, so generated items reuse the
    lesson's actual concepts and difficulty.
    """
    lines: list[str] = []
    topic = (content.topic or content.title or "").strip()
    if topic:
        lines.append(f"Lesson topic: {topic}")
    if content.objectives:
        lines.append("Learning objectives:")
        lines += [f"- {o}" for o in content.objectives if str(o).strip()]
    headings = [s.heading.strip() for s in content.sections
                if getattr(s, "heading", "") and s.heading.strip()]
    if headings:
        lines.append("Sections covered: " + "; ".join(headings))
    examples = []
    for s in content.sections:
        for ex in getattr(s, "examples", []) or []:
            prob = (ex.problem_latex or "").strip()
            ans = (ex.final_answer_latex or "").strip()
            if prob:
                examples.append(f"- {prob}" + (f"  [answer: {ans}]" if ans else ""))
    if examples:
        lines.append("Worked examples in this lesson:")
        lines += examples
    practice = []
    for s in content.sections:
        if getattr(s, "kind", "") == "practice":
            for m in (s.math_latex or []):
                if str(m).strip():
                    practice.append(f"- {m.strip()}")
            body = (s.body_markdown or "").strip()
            if body and not practice:
                practice.append(body)
    if practice:
        lines.append("Practice problems from the lesson:")
        lines += practice
    return "\n".join(lines).strip()


def lesson_content_from_json(raw: str) -> LessonContent:
    """Parse stored JSON into LessonContent; corrupt data -> empty lesson."""
    import json

    try:
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            return LessonContent()
        return LessonContent.model_validate(data)
    except Exception:
        return LessonContent()


def assignment_items_from_json(raw: str) -> list[AssignmentItem]:
    """Parse stored JSON into AssignmentItems; corrupt data -> empty list."""
    import json

    try:
        data = json.loads(raw or "[]")
        if not isinstance(data, list):
            return []
        return [AssignmentItem.model_validate(d) for d in data
                if isinstance(d, dict)]
    except Exception:
        return []


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
    solution via verifier.verify_canonical. If the problem cannot be
    canonically solved from prompt alone (unverifiable), deterministic step
    consistency (final answer matches the last step) is verified instead.
    Returns (ok, reasons).
    """
    from .verifier import verify_canonical, check_equivalent

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
    if verdict.status == "correct":
        return True, reasons
    if verdict.status == "incorrect":
        reasons.append(f"canonical check: {verdict.detail}")
        return False, reasons
    # If the problem cannot be solved canonically from prompt alone ("unverifiable",
    # e.g. slope from points, line equations, function evaluations, word problems),
    # verify that the final answer is deterministically equivalent to the last step.
    if question.final_answer_latex and question.steps:
        eq_verdict = check_equivalent(
            question.final_answer_latex,
            question.steps[-1].expected_input_latex,
        )
        if eq_verdict.status == "correct":
            return True, reasons
        reasons.append(
            f"final answer not equivalent to last step: {eq_verdict.detail}"
        )
        return False, reasons
    reasons.append(f"canonical check: {verdict.detail}")
    return False, reasons
