"""Opt-in LIVE Nebius smoke set for the floating tutor (Phase A).

Skipped by default. Run once before the demo with:
    py -m pytest tests/test_float_live.py -q -s --live

Each test reports pass/fail, latency, and a rough per-call cost note.
Requires NEBIUS_API_KEY and DEMO_MODE=false. No mocks here.
"""
import os
import time

import pytest

from app import nebius
from app import float_plan as plan
from app.verifier import normalize_math, _parse_single
from app import config

LIVE = os.getenv("LIVE") == "1" or "--live" in os.sys.argv
NEEDS_KEY = pytest.mark.skipif(
    not (config.NEBIUS_API_KEY and not config.DEMO_MODE) or not LIVE,
    reason="live Nebius smoke: run with --live and a configured key",
)

COST_NOTE = ("Nebius Token Factory billing is per-token; each case below "
             "makes 1-3 chat calls at <=1200 max_tokens on the math model.")


def _timed(label, fn, *args):
    t0 = time.time()
    try:
        out = fn(*args)
        ok = out is not None
    except Exception as exc:  # noqa: BLE001 - report, don't crash the set
        return (label, False, time.time() - t0, f"EXC: {exc}")
    return (label, ok, time.time() - t0, COST_NOTE)


@NEEDS_KEY
def test_live_trig_plan():
    p, src = plan.build_plan("2cos^2 x - 3cos x + 1 = 0 in [0, 2*pi)",
                             allow_llm=True)
    assert p and src == "trig_quad", (p, src)
    print(f"\ntrig plan: src={src}")


@NEEDS_KEY
def test_live_limit_plan():
    p, src = plan.build_plan(
        "limit as x->2 of (x^2 - 4)/(x - 2)", allow_llm=True)
    assert p and src == "limit", (p, src)
    ok, reason = plan.validate_plan(p, "limit as x->2 of (x^2 - 4)/(x - 2)")
    assert ok, reason
    print(f"\nlimit plan validated: {reason}")


@NEEDS_KEY
def test_live_derivative_plan():
    p, src = plan.build_plan("differentiate x^3*sin(x)", allow_llm=True)
    assert p and src == "derivative", (p, src)
    print(f"\nderivative plan: {len(p)} steps")


@NEEDS_KEY
def test_live_normalizer():
    t0 = time.time()
    out = nebius.normalize_via_llm("two cosine squared of x minus three cosine x plus one equals zero")
    dt = time.time() - t0
    print(f"\nnormalizer ({dt:.1f}s): {out!r}")
    assert out is not None
    parsed = _parse_single(out)
    assert parsed is not None


@NEEDS_KEY
def test_live_conceptual_two_pass():
    t0 = time.time()
    v = nebius.judge_conceptual(
        "interval choice", "[0, 2*pi)",
        "only solutions in [0, 2*pi)")
    dt = time.time() - t0
    print(f"\nconceptual two-pass ({dt:.1f}s): {v}")
    assert v in ("correct", "incorrect", "unclear")


@NEEDS_KEY
def test_live_llm_plan_uncovered():
    t0 = time.time()
    p, src = plan.build_plan("solve |x - 3| = 5", allow_llm=True)
    dt = time.time() - t0
    print(f"\nabs-value plan ({dt:.1f}s): src={src}")
    assert p is not None or src == "none"
