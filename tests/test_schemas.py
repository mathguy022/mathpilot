"""Tests for the Phase 1 authoring schemas + Activity adapter."""
from app import schemas


def _sample_question() -> schemas.Question:
    return schemas.Question(
        id=7,
        prompt_latex="Solve $2(x+3)=14$.",
        difficulty=2,
        steps=[
            schemas.QuestionStep(
                step_id="divide",
                description="Divide both sides by 2.",
                expected_input_latex="x+3=7",
                accepted_equivalents=["x + 3 = 7"],
                hints=["What undoes the multiplication?"],
                objective_tags=["Inverse Operations"],
            )
        ],
        final_answer_latex="x+3=7",
        policy=schemas.TutoringPolicy(
            allow_solution_reveal=False,
            max_attempts_before_reveal=2,
            accepted_methods="Inverse operations.",
        ),
    )


def test_step_legacy_roundtrip():
    step = _sample_question().steps[0]
    legacy = schemas.step_to_legacy(step)
    assert legacy["prompt"] == "Divide both sides by 2."
    assert legacy["expected"] == "x+3=7"
    back = schemas.step_from_legacy(legacy, 0)
    assert back.description == step.description
    assert back.expected_input_latex == step.expected_input_latex
    assert back.hints == step.hints
    assert back.accepted_equivalents == step.accepted_equivalents
    assert back.objective_tags == step.objective_tags


def test_question_activity_fields_preserve_policy():
    q = _sample_question()
    fields = schemas.question_to_activity_fields(q)
    assert fields["problem_statement"] == "Solve $2(x+3)=14$."
    assert fields["allow_solution_reveal"] is False
    assert fields["max_attempts_before_reveal"] == 2
    assert fields["accepted_methods"] == "Inverse operations."


def test_question_from_activity_roundtrip():
    class FakeActivity:
        id = 7
        problem_statement = "Solve $2(x+3)=14$."
        expected_steps_json = (
            '[{"prompt": "Divide both sides by 2.", "expected": "x+3=7", '
            '"hint": ["What undoes the multiplication?"], '
            '"accepted_equivalents": ["x + 3 = 7"], '
            '"objective_tags": ["Inverse Operations"]}]'
        )
        accepted_methods = "Inverse operations."
        allow_solution_reveal = False
        max_attempts_before_reveal = 2

    q = schemas.question_from_activity(FakeActivity())
    assert q.id == 7
    assert len(q.steps) == 1
    assert q.steps[0].expected_input_latex == "x+3=7"
    assert q.policy.allow_solution_reveal is False
    assert q.policy.max_attempts_before_reveal == 2


def test_verify_question_passes_for_good_content():
    ok, reasons = schemas.verify_question(_sample_question())
    assert ok, reasons


def test_verify_question_fails_for_bad_content():
    q = _sample_question()
    q.final_answer_latex = "x=999"
    ok, reasons = schemas.verify_question(q)
    assert not ok
    assert reasons


def test_verify_question_accepts_multi_answer_step():
    q = _sample_question()
    q.steps[0].expected_input_latex = "x=2, x=3"
    q.final_answer_latex = "x=3, x=2"
    ok, reasons = schemas.verify_question(q)
    assert ok, reasons


def test_split_answers_respects_nesting():
    assert schemas.split_answers("x=2, x=3") == ["x=2", "x=3"]
    assert schemas.split_answers(r"x=\frac{-7 \pm \sqrt{25}}{4}") == [
        r"x=\frac{-7 \pm \sqrt{25}}{4}"]
