"""Nebius Token Factory adapter (OpenAI-compatible chat completions).

Two-stage pipeline:
  1. MATH_MODEL (Qwen) judges mathematical correctness of the step.
  2. SOCRATIC_MODEL (Nemotron) crafts Socratic feedback from that verdict.
Policy enforcement stays in application code (tutor.py), never in the prompt.
"""
from __future__ import annotations

import httpx
from . import config

TIMEOUT_S = 30.0
MAX_CTX_TURNS = 6


def _chat(model: str, messages: list[dict], max_tokens: int = 400) -> str:
    if not config.ai_configured():
        raise RuntimeError("AI not configured (missing key or DEMO_MODE=true)")
    url = config.NEBIUS_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {config.NEBIUS_API_KEY}"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.3, "reasoning_effort": "none",
               "chat_template_kwargs": {"enable_thinking": False}}
    with httpx.Client(timeout=TIMEOUT_S) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    try:
        choice = data["choices"][0]["message"]
        # Some reasoning models put text in `reasoning_content`, keep final `content`.
        content = choice.get("content") or ""
        if not content.strip() and choice.get("reasoning_content"):
            content = choice["reasoning_content"]
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
