"""Tests for Phase 2 generation helpers (no live API calls)."""
from app import generate


def test_bound_notes_truncates():
    long_text = "x" * (generate.MAX_NOTES_CHARS + 100)
    bounded, truncated = generate.bound_notes(long_text)
    assert truncated is True
    assert len(bounded) == generate.MAX_NOTES_CHARS


def test_bound_notes_short_passthrough():
    bounded, truncated = generate.bound_notes("Solve x+1=2.")
    assert truncated is False
    assert bounded == "Solve x+1=2."


def test_strip_fences():
    fenced = "```json\n[1, 2]\n```"
    assert generate.strip_fences(fenced) == "[1, 2]"
    assert generate.strip_fences("[1, 2]") == "[1, 2]"


def test_sanitize_expected():
    assert generate.sanitize_expected("x+3=7") == "x+3=7"
    assert generate.sanitize_expected("Answer: x=2") == "x=2"
    assert generate.sanitize_expected("x=2; check by substitution") == "x=2"
    assert generate.sanitize_expected("x=2 (check: plug back in)") == "x=2"
    assert generate.sanitize_expected("First multiply both sides out") == ""
    assert generate.sanitize_expected("") == ""
    assert generate.sanitize_expected(
        "x-2=0 \\text{ or } x-3=0") == "x-2=0, x-3=0"
    assert generate.sanitize_expected(
        "x-2=0 \\\\text{ or } x-3=0") == "x-2=0, x-3=0"


def test_parse_candidates_rejects_non_list():
    try:
        generate.parse_candidates('{"a": 1}')
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_candidate_to_question_mapping():
    candidate = {
        "prompt": "Solve $2(x+3)=14$.",
        "answer": "x+3=7",
        "objective": "Inverse Operations",
        "difficulty": 2,
        "steps": [{"description": "Divide by 2.", "expected": "x+3=7"}],
    }
    q = generate.candidate_to_question(candidate)
    assert q.prompt_latex == "Solve $2(x+3)=14$."
    assert q.final_answer_latex == "x+3=7"
    assert q.source == "ai-generated"
    assert q.is_verified is False
    assert q.difficulty == 2
    assert len(q.steps) == 1


def test_generate_requires_ai(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    monkeypatch.setenv("DEMO_MODE", "true")
    import importlib
    import app.config as config
    importlib.reload(config)
    try:
        generate.generate_questions("notes", "Algebra", 1)
    except RuntimeError as exc:
        assert "not configured" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    finally:
        importlib.reload(config)


def test_generate_verifies_candidates(monkeypatch):
    import app.generate as gen

    fake_output = (
        '[{"prompt": "Solve $x+1=2$.", "answer": "x=1", '
        '"objective": "Isolate", "difficulty": 1, '
        '"steps": [{"description": "Subtract 1.", "expected": "x=1"}]}, '
        '{"prompt": "Bad", "answer": "??? hello", '
        '"objective": "Bad", "difficulty": 9, '
        '"steps": [{"description": "???", "expected": "??? hello"}]}]'
    )
    monkeypatch.setattr(gen.nebius, "_chat", lambda *a, **k: fake_output)
    monkeypatch.setattr(gen.config, "ai_configured", lambda: True)
    result = gen.generate_questions("notes", "Algebra", 2)
    assert len(result["verified"]) == 1
    assert result["verified"][0]["is_verified"] is True
    assert result["discarded"] == 1
    assert result["discard_reasons"]
