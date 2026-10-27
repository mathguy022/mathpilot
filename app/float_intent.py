"""Rule-based intent router for the floating tutor (Student view only).

Runs before grading. No DB changes, no LLM calls here.
"""
from __future__ import annotations

import re

_NEW_LEAD = re.compile(
    r"^\s*(solve|find|simplify|evaluate|factor|differentiate|integrate|prove)\b",
    re.IGNORECASE,
)
_HELP = ("stuck", "hint", "hint?", "help", "i don't know", "idk", "lost")
# Any prose that names a hint is asking for help, even when the wording is
# not in the exact-phrase list ("another hint please", "hint 2?").
_HINT_WORD = re.compile(r"\bhints?\b")
_HELP_PHRASES = (
    "i'm stuck", "im stuck", "i am stuck", "i'm lost", "im lost",
    "i am lost", "no idea", "i don't get", "i dont get", "don't get it",
    "confused", "i give up", "this is hard", "too hard", "help me",
    "i don't know how", "i dont know how", "can't start", "cannot start",
)
_ANSWER_REQ = (
    "give me the answer", "what is the answer", "what's the answer",
    "just tell me", "just solve it", "solve it for me",
    "show me the solution", "reveal the answer", "do it for me",
    "ignore previous instructions", "ignore all instructions",
    "disregard instructions", "override",
)
_WHY = ("why do we", "why do i", "why subtract", "why add",
        "why divide", "why multiply", "why do you", "what does",
        "what do you mean", "explain")
_CONCEPT_ASKS = (
    "what is", "what are", "what's", "whats", "how do", "how to",
    "how does", "when do", "when we", "difference", "meaning of",
    "mean by", "do i", "should i", "can you", "formula for", "rule for",
)
_GREETINGS = (
    "hi", "hii", "hiii", "hello", "hey", "heyy", "yo", "hiya", "howdy",
    "good morning", "good afternoon", "good evening", "sup",
    "thanks", "thank you", "thx", "ty", "ok", "okay", "cool", "nice",
    "great", "yes", "no", "yep", "nope", "sure", "sorry", "hmm", "oh",
    "wow", "lol", "test", "are you there", "who are you", "what are you",
    "your name", "are you real", "are you a bot",
)
_NON_MATH = (
    "president", "election", "football", "recipe", "movie", "song",
    "weather", "politics", "grade me", "gpa", "essay", "translate",
    "python", "java", "coding", "programming", "history", "biology",
    "chemistry", "physics", "english", "spanish", "netflix", "tiktok",
    "instagram", "joke", "love", "relationship", "shopping",
)
# Conversational lead-ins students put in front of a real answer.
_ANSWER_LEAD = re.compile(
    r"^\s*(?:so|then|thus|i\s+get|i\s+got|i\s+think|i\s+have|we\s+get|"
    r"the\s+answer\s+is|my\s+answer\s+is|equals?|answer\s*[:\-])\s*",
    re.IGNORECASE,
)
_MATH_CHARS = re.compile(
    r"^[\s0-9a-zA-Z+\-*/^=<>!()\[\]{}.,:_\\$%πθ∞√×÷≤≥±°−•]+$"
)
_MATH_WORD = re.compile(
    r"(sin|cos|tan|sec|csc|cot|log|ln|sqrt|lim|derivativ|integral|pi|"
    r"theta|infinity|dx|dy|sum|abs|arcsin|arccos|arctan)",
    re.IGNORECASE,
)
_MATH_TOPIC = (
    "equation", "function", "graph", "slope", "factor", "derivativ",
    "integral", "limit", "fraction", "variable", "solve", "simplif",
    "exponent", "polynomial", "quadratic", "inequality", "angle",
    "radian", "degree", "numerator", "denominator", "coefficient",
    "expression", "term", "isolate", "balance", "percent", "ratio",
    "decimal", "proportion", "slope", "intercept", "vertex", "domain",
    "range", "asymptote",
)


def classify(message: str, has_session: bool) -> str:
    """Return NEW_PROBLEM | STEP_ANSWER | CONCEPT_QUESTION | HELP_REQUEST |
    ANSWER_REQUEST | OFF_TOPIC | CHAT.

    Order matters: the off-topic and question reads come before the
    math-shape test, otherwise prose like "what is a derivative" gets
    graded as a step answer and the student only ever sees a parse error.
    """
    t = (message or "").strip()
    if not t:
        return "HELP_REQUEST"
    low = t.lower()
    if any(p in low for p in _ANSWER_REQ):
        return "ANSWER_REQUEST"
    if low in _HELP or low.strip("!?.") in ("stuck", "hint", "help"):
        return "HELP_REQUEST"
    if (not looks_like_answer(t)
            and (any(p in low for p in _HELP_PHRASES)
                 or _HINT_WORD.search(low))):
        return "HELP_REQUEST"
    if any(w in low for w in _NON_MATH) and not looks_like_answer(t):
        return "OFF_TOPIC"
    if _is_greeting(low):
        return "CHAT"
    # A question about a course topic ("what is a derivative", "why do we
    # subtract") outranks the math-shape test: it is asking, not answering.
    if (not looks_like_answer(t) and "=" not in t
            and (_is_question_form(low) or any(p in low for p in _WHY))
            and _is_math_topic(low)):
        return "CONCEPT_QUESTION"
    if _NEW_LEAD.match(t):
        return "NEW_PROBLEM"
    if looks_like_answer(t):
        if not has_session and "=" in t:
            return "NEW_PROBLEM"
        return "STEP_ANSWER"
    if not has_session and _looks_like_math(t):
        return "NEW_PROBLEM"
    # Unrecognized. With a problem open it is most likely a mistyped or
    # unreadable answer, so let the CAS grade it (unclear -> format help)
    # rather than chat past it. With nothing open, it is conversation.
    return "STEP_ANSWER" if has_session else "CHAT"


def _is_math_topic(low: str) -> bool:
    return bool(_MATH_WORD.search(low)) or any(w in low for w in _MATH_TOPIC)


def _is_question_form(low: str) -> bool:
    return low.rstrip().endswith("?") or bool(re.match(
        r"^\s*(what|whats|what's|how|why|when|which|who|whose|is|are|can|"
        r"could|do|does|did|should|would|explain|define|tell me)\b", low))


def _is_greeting(low: str) -> bool:
    core = low.strip("!?. ,").replace("  ", " ")
    return core in _GREETINGS or core.replace(" ", "") in [
        g.replace(" ", "") for g in _GREETINGS]


def clean_answer(text: str) -> str:
    """Return the math-shaped core of a student's message, or "" for prose.

    Strips conversational lead-ins ("so", "i got", "the answer is") and
    spells out "equals", so "so 3x = 9" grades as "3x = 9". Returns "" when
    what is left reads like sentences, which lets the grader answer
    "unclear" instead of marking a paragraph wrong.
    """
    t = re.sub(r"\b(?:the|my|our|your)\s+answer\s+is\b", " ",
               (text or "").strip(), flags=re.IGNORECASE)
    t = re.sub(r"\bequals?\b", "=", t, flags=re.IGNORECASE)
    prev = None
    while prev != t:  # stacked lead-ins: "so then i got 9 = 3x"
        prev = t
        t = _ANSWER_LEAD.sub("", prev)
    t = t.strip().strip("$").strip()
    if not t or not _MATH_CHARS.match(t):
        return ""
    if re.search(r"[a-zA-Z]{5,}", t) and not _MATH_WORD.search(t):
        return ""  # long alphabetic runs are words, not variables
    if not re.search(r"\d|[=<>+\-*/^√×÷−]|\\frac|\\sqrt", t):
        return ""
    return t


def looks_like_answer(text: str) -> bool:
    """True when the message reads as a math response rather than prose."""
    return bool(clean_answer(text))


def _looks_like_math(t: str) -> bool:
    if looks_like_answer(t):
        return True
    if "=" in t:
        return True
    for kw in ("lim", "d/dx", "derivative", "integral", "sin", "cos",
               "tan", "log", "ln", "sqrt", "pi", "^", "dx"):
        if kw in t.lower():
            return True
    return False


def looks_like_new_problem_mid_session(text: str) -> bool:
    """Heuristic: mid-session message that is likely a new problem."""
    t = (text or "").strip()
    if _NEW_LEAD.match(t):
        return True
    # Equation much longer/different shape than a step answer is handled
    # by the caller comparing against the plan; this is just the verb-less
    # complete-equation signal (e.g. pasted "(pi/4)x = -1").
    if "=" in t and len(t) > 4:
        # Caller decides: if it parses as a full problem AND differs from
        # the expected step shape, ask for confirmation.
        return True
    return False
