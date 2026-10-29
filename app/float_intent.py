"""Rule-based intent router for the floating tutor (Student view only).

Runs before grading. No DB changes, no LLM calls here.
"""
from __future__ import annotations

import re

_NEW_LEAD = re.compile(
    r"^\s*(solve|find|simplify|evaluate|factor|differentiate|integrate|prove)\b",
    re.IGNORECASE,
)
_HELP = ("stuck", "hint", "hint?", "help", "lost")
# Any prose that names a hint is asking for help, even when the wording is
# not in the exact-phrase list ("another hint please", "hint 2?").
_HINT_WORD = re.compile(r"\bhints?\b")
_HELP_PHRASES = (
    "i'm stuck", "im stuck", "i am stuck", "i'm lost", "im lost",
    "i am lost", "no idea", "i don't get", "i dont get", "don't get it",
    "confused", "i don't know how", "i dont know how", "can't start",
    "cannot start", "i need help", "need a hand", "i dont understand",
    "i don't understand", "im confused", "i am confused", "help me please",
)
# "Where do I start?" is not a hint request: the student needs Step 1, not a
# rung. Apostrophe-free spellings and the common "I do know" mistype count too.
_START_PHRASES = (
    "what should i do", "what should i do next", "what should i do here",
    "what do i do", "what do i do now", "what do i do here", "what to do",
    "what to do here", "what to do next", "what now", "whats next",
    "what's next", "what is next", "where do i start", "where should i start",
    "where do i begin", "how do i start", "how do i begin", "how do i start it",
    "i do know", "i dont know", "i don know", "i do not know", "idk",
    "no clue", "i have nothing", "im blank", "i am blank", "mind is blank",
    "start over", "begin again", "not sure what to do", "unsure what to do",
    "tell me the first step", "whats the first step", "what is the first step",
    "first step", "help me", "help me out", "show me how to start",
)
# Frustration outranks a hint request: the student gets acknowledgement and a
# smaller step, not another rung.
_FRUSTRATED = (
    "i give up", "im giving up", "i am giving up", "gave up", "i quit",
    "this is stupid", "this is dumb", "stupid math", "i hate this",
    "i hate math", "hate this", "i cant do this", "i can not do this",
    "this is hopeless", "hopeless", "pointless", "waste of time",
    "too hard for me", "im too dumb", "i am too dumb", "i m not smart",
    "why is this so hard", "i m done", "im done with this", "never mind",
    "this is too hard", "this is hard", "too hard", "too difficult",
    "cant take this", "i'm frustrated",
    "im frustrated", "so frustrated", "annoying", "i m angry", "so angry",
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
# Operation verbs are course vocabulary too: "why do we subtract?" is a
# concept question, not a request for Step 1. Word-bounded, because a
# substring read of "add" would match "address".
_MATH_VERB = re.compile(
    r"\b(subtract|add|divide|multipl|cancel|combin|expand|square|cube|"
    r"root|substitut|isolat|simplifi|solv|evalu|differenti|integrat|"
    r"move|shift|swap|swap sides|bring|get rid|cancel out|"
    r"sine|cosine|tangent|secant|cosecant|cotangent|log|antiderivative|"
    r"completing|discriminant|facto)\w*\b",
    re.IGNORECASE,
)


def classify(message: str, has_session: bool) -> str:
    """Return NEW_PROBLEM | STEP_ANSWER | CONCEPT_QUESTION | HELP_REQUEST |
    START_STEP | FRUSTRATED | ANSWER_REQUEST | OFF_TOPIC | CHAT.

    Order matters: the off-topic and question reads come before the
    math-shape test, otherwise prose like "what is a derivative" gets
    graded as a step answer and the student only ever sees a parse error.
    """
    t = (message or "").strip()
    if not t:
        return "HELP_REQUEST"
    low = t.lower()
    # Students type help phrases with and without apostrophes ("i dont know").
    bare = low.replace("'", "").replace("\u2019", "")
    if any(p in low for p in _ANSWER_REQ):
        return "ANSWER_REQUEST"
    if not looks_like_answer(t) and any(p in bare for p in _FRUSTRATED):
        return "FRUSTRATED"
    if low in _HELP or low.strip("!?.") in ("stuck", "hint", "help"):
        return "HELP_REQUEST"
    if (not looks_like_answer(t)
            and (any(p in bare for p in _HELP_PHRASES)
                 or _HINT_WORD.search(low))):
        return "HELP_REQUEST"
    if (not looks_like_answer(t)
            and any(p in bare for p in _START_PHRASES)):
        return "START_STEP"
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
    # A question that carries no math at all is asking for a direction, so it
    # goes to the step, not to the grader: "how do I even approach this?" must
    # never read back as "I couldn't match that to this item".
    if (not _is_gibberish(low) and not _looks_like_math(t)
            and (low.endswith("?") or _is_question_form(low))):
        return "START_STEP" if has_session else "CHAT"
    # Unrecognized. With a problem open it is most likely a mistyped or
    # unreadable answer, so let the CAS grade it (unclear -> format help)
    # rather than chat past it. With nothing open, it is conversation.
    return "STEP_ANSWER" if has_session else "CHAT"


def _is_math_topic(low: str) -> bool:
    return (bool(_MATH_WORD.search(low)) or bool(_MATH_VERB.search(low))
            or any(w in low for w in _MATH_TOPIC))


def _is_gibberish(low: str) -> bool:
    """True for keyboard noise ("asdfgh jkl"), which is an unreadable answer.

    Real help phrases are made of vowels-and-consonants words; a bare run of
    three consonants is not English and not math, so it must still reach the
    checker and get format help instead of being read as "I'm lost".
    """
    for token in re.findall(r"[a-z]{3,}", low or ""):
        if not re.search(r"[aeiouy]", token):
            return True
    return False


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
