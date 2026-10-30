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


def test_sanitize_expected_keeps_trailing_math_group():
    # A trailing parenthetical is usually real math, not commentary: stripping
    # it silently changed the expected answer or cut the expression in half.
    assert generate.sanitize_expected(
        "(x+2)(x-3)") == "(x+2)(x-3)"
    assert generate.sanitize_expected(
        "(2x^2 - x^2) + (3x + 5x) + (-4 + 2)"
    ) == "(2x^2 - x^2) + (3x + 5x) + (-4 + 2)"
    assert generate.sanitize_expected(
        "x^3 - 6x^2 + 11x - 6 = (x-1)(x^2 - 5x + 6)"
    ) == "x^3 - 6x^2 + 11x - 6 = (x-1)(x^2 - 5x + 6)"
    # Prose parentheticals are still dropped.
    assert generate.sanitize_expected("x=2 (verify by substitution)") == "x=2"


def test_sanitize_expected_drops_truncated_expression():
    # A trailing operator means the model output was cut mid-expression; the
    # step must not become an expectation that grades a truncated answer.
    assert generate.sanitize_expected("(3x + 5x) +") == ""
    assert generate.sanitize_expected("x^2 -") == ""


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


def test_shortfall_retries_after_one_empty_round(monkeypatch):
    """One empty top-up round must not end the run.

    Asking for 2 questions and stopping after a single fruitless round was
    how a request for 5 arrived as 1 item.
    """
    import app.generate as gen

    good_one = (
        '[{"prompt": "Solve $x+1=2$.", "answer": "x=1", '
        '"objective": "Isolate", "difficulty": 1, '
        '"steps": [{"description": "Subtract 1.", "expected": "x=1"}]}]'
    )
    good_two = (
        '[{"prompt": "Solve $x+2=3$.", "answer": "x=1", '
        '"objective": "Isolate", "difficulty": 1, '
        '"steps": [{"description": "Subtract 2.", "expected": "x=1"}]}]'
    )
    bad = (
        '[{"prompt": "Bad", "answer": "??? hello", '
        '"objective": "Bad", "difficulty": 9, '
        '"steps": [{"description": "???", "expected": "??? hello"}]}]'
    )
    calls = {"n": 0}

    def fake_chat(*args, **kwargs):
        calls["n"] += 1
        return {1: good_one, 2: bad, 3: good_two}.get(calls["n"], bad)

    monkeypatch.setattr(gen.nebius, "_chat", fake_chat)
    monkeypatch.setattr(gen.config, "ai_configured", lambda: True)
    result = gen.generate_questions("notes", "Algebra", 2)
    assert calls["n"] == 3          # one empty round was retried, not fatal
    assert len(result["verified"]) == 2
    assert result["shortfall"] == 0


def test_malformed_json_is_retried(monkeypatch):
    """A junk reply must not end the request with zero questions."""
    import app.generate as gen

    good = (
        '[{"prompt": "Solve $x+1=2$.", "answer": "x=1", '
        '"objective": "Isolate", "difficulty": 1, '
        '"steps": [{"description": "Subtract 1.", "expected": "x=1"}]}]'
    )
    replies = ["Sorry, here are some problems: ...", good]
    calls = {"n": 0}

    def fake_chat(*args, **kwargs):
        calls["n"] += 1
        return replies[min(calls["n"], len(replies)) - 1]

    monkeypatch.setattr(gen.nebius, "_chat", fake_chat)
    monkeypatch.setattr(gen.config, "ai_configured", lambda: True)
    result = gen.generate_questions("notes", "Algebra", 1)
    assert calls["n"] == 2
    assert len(result["verified"]) == 1


def test_unparseable_after_retry_reports_full_shape(monkeypatch):
    import app.generate as gen

    monkeypatch.setattr(gen.nebius, "_chat", lambda *a, **k: "not json at all")
    monkeypatch.setattr(gen.config, "ai_configured", lambda: True)
    result = gen.generate_questions("notes", "Algebra", 3)
    assert result["verified"] == []
    assert result["requested"] == 3
    assert result["delivered"] == 0
    assert result["shortfall"] == 3
    assert result["discard_reasons"]


def test_transient_provider_error_is_retried(monkeypatch):
    """A dropped connection must not surface as an instant empty result."""
    import app.generate as gen

    good = (
        '[{"prompt": "Solve $x+1=2$.", "answer": "x=1", '
        '"objective": "Isolate", "difficulty": 1, '
        '"steps": [{"description": "Subtract 1.", "expected": "x=1"}]}]'
    )
    calls = {"n": 0}

    def flaky_chat(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("connection reset by peer")
        return good

    monkeypatch.setattr(gen.nebius, "_chat", flaky_chat)
    monkeypatch.setattr(gen.config, "ai_configured", lambda: True)
    monkeypatch.setattr(gen.time, "sleep", lambda _s: None)
    result = gen.generate_questions("notes", "Algebra", 1)
    assert calls["n"] == 3
    assert len(result["verified"]) == 1


def test_provider_failure_after_retries_is_provider_unavailable(monkeypatch):
    import app.generate as gen

    def always_fail(*args, **kwargs):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(gen.nebius, "_chat", always_fail)
    monkeypatch.setattr(gen.config, "ai_configured", lambda: True)
    monkeypatch.setattr(gen.time, "sleep", lambda _s: None)
    try:
        gen.generate_questions("notes", "Algebra", 2)
    except gen.ProviderUnavailable:
        pass
    else:  # pragma: no cover - the provider never returned a reply
        raise AssertionError("expected ProviderUnavailable")


def test_broadening_fallback_fills_count_when_lesson_notes_are_conceptual(monkeypatch):
    """A conceptual lesson must not leave the instructor with 1 of 5 items."""
    import app.generate as gen

    bad = ('[{"prompt": "Describe the end behavior of the graph.", '
           '"answer": "up on the left, down on the right", '
           '"objective": "polynomial", "difficulty": 2, '
           '"steps": [{"description": "Describe.", '
           '"expected": "end behavior prose"}]}]')
    good = ('[{"prompt": "Solve $x+1=2$.", "answer": "x=1", '
            '"objective": "Isolate", "difficulty": 1, '
            '"steps": [{"description": "Subtract 1.", "expected": "x=1"}]}]')
    seen_broadening = {"hit": False}

    def fake_chat(model, messages, **kwargs):
        text = " ".join(str(m.get("content", "")) for m in messages)
        # The broadened round drops the lesson notes entirely.
        if "sketch polynomials" not in text:
            seen_broadening["hit"] = True
            return good
        return bad

    monkeypatch.setattr(gen.nebius, "_chat", fake_chat)
    monkeypatch.setattr(gen.config, "ai_configured", lambda: True)
    result = gen.generate_questions("Lesson: sketch polynomials.", "Polynomials",
                                    1, allow_broadening=True)
    assert seen_broadening["hit"] is True
    assert len(result["verified"]) == 1
    assert result["broadened"] is True


def test_broadening_is_opt_in(monkeypatch):
    import app.generate as gen

    bad = ('[{"prompt": "Describe the graph.", "answer": "prose answer", '
           '"objective": "polynomial", "difficulty": 2, '
           '"steps": [{"description": "Describe.", "expected": "prose"}]}]')
    monkeypatch.setattr(gen.nebius, "_chat", lambda *a, **k: bad)
    monkeypatch.setattr(gen.config, "ai_configured", lambda: True)
    result = gen.generate_questions("notes", "Polynomials", 2)
    assert result["verified"] == []
    assert result["broadened"] is False


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
    # The gate rejected the malformed candidate; the top-up loop may also
    # re-surface it as a duplicate, so assert at least one discard is logged.
    assert result["discarded"] >= 1
    assert result["discard_reasons"]
