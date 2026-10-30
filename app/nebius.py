"""Nebius Token Factory adapter (OpenAI-compatible chat completions).

Two-stage pipeline:
  1. MATH_MODEL (Qwen) judges mathematical correctness of the step.
  2. SOCRATIC_MODEL (Nemotron) crafts Socratic feedback from that verdict.
Policy enforcement stays in application code (tutor.py), never in the prompt.
"""
from __future__ import annotations

import httpx
import logging

from . import config

TIMEOUT_S = 30.0
# The widget aborts at 20s, so conversational replies must give up sooner
# and let the caller fall back to its deterministic wording.
CHAT_TIMEOUT_S = 12.0
MAX_CTX_TURNS = 6


def _chat(model: str, messages: list[dict], max_tokens: int = 400,
          timeout_s: float = TIMEOUT_S) -> str:
    if not config.ai_configured():
        raise RuntimeError("AI not configured (missing key or DEMO_MODE=true)")
    url = config.NEBIUS_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {config.NEBIUS_API_KEY}"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.3, "reasoning_effort": "none",
               "chat_template_kwargs": {"enable_thinking": False}}
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    try:
        choice = data["choices"][0]["message"]
        # Some reasoning models put text in `reasoning_content` or `reasoning`, keep final `content`.
        content = choice.get("content") or ""
        if not content.strip():
            content = choice.get("reasoning_content") or choice.get("reasoning") or ""
        content = content.strip()
        if not content:
            raise ValueError("empty model reply")
        return content
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Nebius response: {exc}") from exc


MATH_JUDGE_SYSTEM = (
    "You are a precise mathematics checker. Given a problem step, the expected "
    "answer, and the student's response, reply with exactly one line: "
    "CORRECT, INCORRECT, INCOMPLETE, or UNVERIFIABLE, followed by a one-sentence reason. "
    "Never reveal the full solution; only judge this step."
)

MATH_LATEX_RULE = (
    "Write ALL mathematics in LaTeX. Use $...$ for inline math and "
    "$$...$$ for display math. Never write math as plain text such as "
    "x^2 or sqrt(x). Write currency as \\$ so it is not confused with math."
)

SOCRATIC_SYSTEM = (
    "You are MathPilot, a Socratic mathematics tutor. Output ONLY the message the "
    "student will read — never include thinking, analysis, plans, or numbered "
    "reasoning steps. Rules:\n"
    "1. NEVER give the final answer or a complete worked solution unless the activity "
    "policy explicitly allows a reveal and the student has exhausted attempts.\n"
    "2. Ask ONE guiding question at a time; build on the student's actual response.\n"
    "3. Use the deterministic verification verdict provided — do not contradict it.\n"
    "4. Match the allowed hint level: L1 conceptual question, L2 specific hint, "
    "L3 formula/operation guidance, L4 worked substep (only if permitted).\n"
    "5. Be encouraging and concise (under 120 words). " + MATH_LATEX_RULE + "\n"
    "6. If the input is ambiguous or unsupported, ask for clarification."
)


PLAN_SYSTEM = (
    "You are a mathematics curriculum planner. Given a student problem, "
    "produce a hidden step plan as JSON ONLY: "
    '{"topic": "...", "problem": "...", "domain_note": "...", '
    '"steps": [{"instruction": "...", "expected": "...", '
    '"check_type": "...", "reason": "...", "micro_steps": [...]}]}. '
    "Rules: 3-8 small steps, one idea per step. Instructions describe the "
    "ACTION, never the result or the answer. check_type is one of: "
    "equivalent_equation, equivalent_expression, factored_form, "
    "solution_set, numeric_value, derivative, integral, limit, "
    "identity_step, conceptual. Expected values must be CAS-parsable "
    "(SymPy syntax: ** for powers, cos(x) with parentheses, pi for π). "
    "Never include a full worked solution outside the expected fields."
    " Every step must be a DIFFERENT action: never repeat a step, and never "
    "write two steps that say the same thing in different words. If the "
    "task is not 'solve for x' (finding a domain, a range, a quadrant, a "
    "transformation), the steps must do THAT task, and the final step uses "
    "check_type 'conceptual' with the answer written plainly, such as "
    "'x >= 4'. Do not repeat the problem statement as an instruction."
)

NORMALIZER_SYSTEM = (
    "You are a math notation translator. Rewrite the student's message as "
    "clean SymPy-parsable ASCII math and reply with ONLY that expression, "
    "no explanation. Use ** for powers, function parentheses "
    "(cos(x), sin(2*x)), pi for π, sqrt() for roots, Abs() for absolute "
    "value. If the input is not math at all, reply exactly: NOT_MATH."
)

CONCEPT_SYSTEM = (
    "You are a precise mathematics checker. Given a conceptual step "
    "(word problem, proof step, explanation), reply with exactly one line: "
    "CORRECT, INCORRECT, or UNVERIFIABLE, followed by a one-sentence "
    "reason. Never reveal the full solution; only judge this step."
)


def generate_plan(problem: str, task: str = "") -> dict | None:
    """LLM plan fallback for problems no SymPy template covers.

    Returns the parsed plan dict or None. The caller must validate the
    plan with float_plan.validate_plan before use — never trusted blind.
    The task (what the student was actually asked to do) is passed with it:
    without it the plan answers a different question than the one set.
    """
    import json as _json

    ask = f"Problem: {problem}"
    if task:
        ask += f"\nWhat the student must do with it: {task}"
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": ask},
    ]
    try:
        raw = _chat(config.NEBIUS_MATH_MODEL, messages, max_tokens=1200)
    except Exception:
        return None
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        plan = _json.loads(raw[start:end])
    except Exception:
        return None
    if not isinstance(plan, dict) or not isinstance(
            plan.get("steps"), list):
        return None
    steps = [s for s in plan["steps"]
             if isinstance(s, dict) and s.get("instruction")]
    if not (1 <= len(steps) <= 10):
        return None
    normed = []
    for s in steps:
        normed.append({
            "instruction": str(s["instruction"])[:300],
            "expected_expression": str(s.get("expected", ""))[:300],
            "check_type": str(s.get("check_type",
                                   "equivalent_expression"))[:40],
            "reason": str(s.get("reason", ""))[:300],
            "micro_steps": [str(m)[:200] for m in
                            s.get("micro_steps", [])][:4],
        })
    plan["steps"] = normed
    return plan


def normalize_via_llm(text: str) -> str | None:
    """One-shot LLM math normalizer. Returns CAS-syntax or None.

    The caller re-parses the result with SymPy; the math is never
    trusted from the LLM.
    """
    messages = [
        {"role": "system", "content": NORMALIZER_SYSTEM},
        {"role": "user", "content": text[:500]},
    ]
    try:
        out = _chat(config.NEBIUS_MATH_MODEL, messages,
                    max_tokens=200).strip()
    except Exception:
        return None
    if not out or out == "NOT_MATH":
        return None
    return out[:500]


def judge_conceptual(problem: str, expected: str, student: str) -> str:
    """Two independent LLM passes; BOTH must agree CORRECT to pass.

    Returns 'correct' | 'incorrect' | 'unclear'. Used only for steps the
    CAS cannot check (word problems, proof steps, explanations).
    """
    votes = []
    for _ in range(2):
        messages = [
            {"role": "system", "content": CONCEPT_SYSTEM},
            {"role": "user", "content": (
                f"Problem step: {problem}\nExpected: {expected}\n"
                f"Student: {student}"
            )},
        ]
        try:
            out = _chat(config.NEBIUS_MATH_MODEL, messages,
                        max_tokens=150).upper()
        except Exception:
            return "unclear"
        if out.startswith("CORRECT"):
            votes.append(True)
        elif out.startswith("INCORRECT"):
            votes.append(False)
        else:
            return "unclear"
    if all(votes):
        return "correct"
    return "incorrect"


def judge_step(problem: str, expected: str, student: str) -> str:
    messages = [
        {"role": "system", "content": MATH_JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"Problem step: {problem}\nExpected: {expected}\nStudent: {student}"
        )},
    ]
    return _chat(config.NEBIUS_MATH_MODEL, messages, max_tokens=150)


def socratic_reply(context: dict, history: list[dict]) -> str:
    ctx = (
        f"Problem: {context['problem']}\n"
        f"Current step ({context['step_index']+1}/{context['total_steps']}): {context['step']}\n"
        f"Expected: {context['expected']}\n"
        f"Student response: {context['student']}\n"
        f"Deterministic verdict: {context['verdict']} ({context['verdict_detail']})\n"
        f"AI judge: {context.get('ai_judge', 'n/a')}\n"
        f"Allowed hint level: {context['hint_level']}\n"
        f"Solution reveal allowed: {context['allow_reveal']}"
    )
    messages = [{"role": "system", "content": SOCRATIC_SYSTEM},
                {"role": "user", "content": ctx}]
    for turn in history[-MAX_CTX_TURNS:]:
        messages.append(turn)
    return _chat(config.NEBIUS_SOCRATIC_MODEL, messages)


def float_chat(system: str, history: list[dict], user: str,
               max_tokens: int = 220) -> str | None:
    """Conversational reply for the floating tutor. None on any failure.

    Correctness is never decided here — the caller has already run SymPy
    and passes the verdict into `system` as a fact to phrase, not re-judge.
    """
    messages = [{"role": "system", "content": system}]
    for turn in history[-MAX_CTX_TURNS:]:
        role = str(turn.get("role", "") or "")
        # The tutor thread logs "student"/"tutor" with a "text" field, the
        # practice history already uses API roles with "content". Accept both
        # or a non-empty history silently contributes nothing.
        text = str(turn.get("text") or turn.get("content") or "").strip()
        if not text:
            continue
        if role in ("student", "user"):
            messages.append({"role": "user", "content": text[:1000]})
        elif role in ("tutor", "assistant"):
            messages.append({"role": "assistant", "content": text[:1000]})
    messages.append({"role": "user", "content": user[:1000]})
    try:
        out = _chat(config.NEBIUS_SOCRATIC_MODEL, messages,
                    max_tokens=max_tokens, timeout_s=CHAT_TIMEOUT_S)
    except Exception as exc:
        # Silent fallback hid a bad request earlier; keep it diagnosable.
        logging.getLogger("mathpilot.tutor").warning(
            "float_chat failed, using scripted reply: %s", exc)
        return None
    out = (out or "").strip()
    if not out or out.upper().startswith("NOT_MATH"):
        return None
    return out[:900]
