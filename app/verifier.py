"""Deterministic math verification with SymPy.

Never executes user code. Parses a bounded subset of expressions and
reports correct / incorrect / incomplete / unverifiable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

MAX_LEN = 2000  # LaTeX is verbose; length still bounded
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z\s\+\-\*\/\^\(\)\.,=d_\{\}:<>!|]+$")

# Unicode math symbols students may type or paste -> ASCII equivalents.
_UNICODE_MAP = {
    "π": "pi", "⋅": "*", "·": "*", "×": "*", "÷": "/",
    "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "⁰": "^0", "¹": "^1",
    "−": "-", "–": "-", "—": "-",
    "≤": "<=", "≥": ">=", "≠": "!=",
    "∞": "oo",
    "⅓": "(1/3)", "⅔": "(2/3)", "½": "(1/2)", "¼": "(1/4)",
    "√": "sqrt", "θ": "theta",
}

_LATEX_CMDS = {
    r"\pi": "pi", r"\cdot": "*", r"\times": "*", r"\div": "/",
    r"\pm": "+-", r"\left": "", r"\right": "", r"\,": " ",
    r"\;": " ", r"\:": " ", r"\ ": " ", r"\!": "",
    r"\quad": " ", r"\qquad": " ",
}


def normalize_math(text: str) -> str:
    """Convert LaTeX / Unicode math into plain ASCII for SymPy.

    Pure string transforms — no evaluation. Handles \\frac{}{}, \\sqrt{},
    \\pi, \\cdot, superscripts, and common Unicode symbols.
    """
    t = text.strip()
    # Collapse JSON double-escapes first: model output arrives with '\\'
    # for every LaTeX backslash after a JSON round-trip.
    t = t.replace("\\\\", "\\")
    for uni, asc in _UNICODE_MAP.items():
        t = t.replace(uni, asc)
    # \frac{a}{b} -> (a)/(b) and \sqrt{a} -> sqrt(a), alternated until stable:
    # converting roots afterwards left \frac{sqrt(...)}{x} unmatched by the
    # flat-brace pattern, and the later braces->parens pass turned the
    # quotient into a product.
    for _ in range(6):
        new = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", t)
        new = re.sub(r"\\sqrt\[([^\]]+)\]\{([^{}]+)\}", r"(\2)**(1/(\1))", new)
        new = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", new)
        if new == t:
            break
        t = new
    for cmd, rep in _LATEX_CMDS.items():
        t = t.replace(cmd, rep)
    # \text{...} / \mathrm{...} wrappers first: unwrap before braces->parens
    t = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\mathit\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", t)
    # LaTeX grouping braces -> parens; $ delimiters
    t = t.replace("{", "(").replace("}", ")")
    t = t.replace("$", "")
    # \mathrm{d}r, dr, ∂-style differentials: normalize derivative tokens
    t = re.sub(r"\\mathrm\{d\}", "d", t)
    t = re.sub(r"\bd\s+([a-zA-Z])", r"d\1", t)
    # strip any remaining unknown \command (safe: drop, don't execute)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    # trig powers: cos^2 x -> cos(x)**2 ; cos^2(x) -> cos(x)**2.
    # Handles sin/cos/tan/sec/csc/cot (+ a-prefixed inverses). The direct
    # ** form avoids implicit-multiplication splitting 'cos (x)' later.
    t = re.sub(
        r"(?<![a-zA-Z])(a?sin|a?cos|a?tan|sec|csc|cot)\s*\^\s*(\d+)"
        r"(?:\s*\(\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\)|\s+([a-zA-Z][a-zA-Z0-9_]*))",
        lambda m: f"{m.group(1)}({m.group(3) or m.group(4)})**{m.group(2)}",
        t,
    )
    # bare trig/ln/log of a variable: cos x -> cos(x); ln x -> ln(x).
    # Skip names already followed by '(' (e.g. output of the power rule).
    # (?<![a-zA-Z]) not \b: the name may follow a digit ('3cos x').
    t = re.sub(
        r"(?<![a-zA-Z])(sin|cos|tan|sec|csc|cot|asin|acos|atan|ln|log|exp|sqrt)"
        r"(?!\s*\()\s+([a-zA-Z][a-zA-Z0-9_]*)",
        r"\1(\2)",
        t,
    )
    # fused function-argument: sin2x -> sin(2*x).
    t = re.sub(
        r"\b(sin|cos|tan|sec|csc|cot|asin|acos|atan|ln|log|exp)\s*"
        r"(\d+)\s*([a-zA-Z][a-zA-Z0-9_]*)",
        r"\1(\2*\3)",
        t,
    )
    # e^x stays exp-shaped; bare e (Euler) -> E for SymPy.
    t = re.sub(r"(?<![a-zA-Z])e(?![a-zA-Z0-9_(])", "E", t)
    # |x| -> Abs(x) (single pair only; nested bars stay unparseable).
    t = re.sub(r"\|([^|()=<>!]+)\|", r"Abs(\1)", t)
    # separate known constants/functions glued to variables: pir -> pi*r.
    # The negative lookbehind avoids splitting 'sin(' into 'sin*('.
    for const in ("pi", "sqrt", "sin", "cos", "tan", "sec", "csc",
                  "cot", "asin", "acos", "atan", "log", "ln", "exp",
                  "theta", "oo"):
        t = re.sub(rf"(?<![a-zA-Z]){const}(?=[a-zA-Z])", const + "*", t)
    # protect derivative tokens (dr, dt, dx...) from implicit-multiplication split
    t = re.sub(r"\bd([a-zA-Z])\b", r"DERIV\1", t)
    # insert * for juxtaposition the parser misses: )x, 2x, x(, x y
    t = re.sub(r"(\))(?=[a-zA-Z0-9\(])", r"\1*", t)
    t = re.sub(r"(\d)(?=[a-zA-Z\(])", r"\1*", t)
    t = re.sub(r"([a-zA-Z\)\d])\s+([a-zA-Z\(])", r"\1*\2", t)
    t = t.replace("DERIV", "d")
    # restore derivative as its own symbol: dr -> Symbol('dr')
    t = re.sub(r"(?<![a-zA-Z])d([a-zA-Z])(?![a-zA-Z])", r"d_\1", t)
    return t.strip()


@dataclass
class Verdict:
    status: str  # correct|incorrect|incomplete|unverifiable
    detail: str
    normalized: str = ""


def _parse_single(text: str):
    text = text.strip()
    if len(text) > MAX_LEN or not text:
        raise ValueError("input too long or empty")
    if "=" in text:  # compare RHS for equations like V = ...
        text = text.split("=", 1)[1]
    text = normalize_math(text)
    if not text:
        raise ValueError("empty after normalization")
    if not _ALLOWED_RE.match(text):
        bad = sorted(set(text) - set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ +-*/^().,=d_{}"))
        raise ValueError(f"unsupported characters: {bad[:5]}")
    text = text.replace("^", "**")
    return parse_expr(text, transformations=TRANSFORMATIONS)


def _split_top_level(text: str) -> list[str]:
    """Split on top-level commas: 'x=2, x=3' -> ['x=2', 'x=3'].

    Commas inside braces/parens do not split.
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


def _safe_parse(text: str):
    """Parse one expression; multi-answer input parses if every part parses."""
    parts = _split_top_level(text.split("=", 1)[1] if "=" in text else text)
    if len(parts) > 1:
        parsed = [_parse_single(p) for p in parts]
        return parsed[0] if len(parsed) == 1 else tuple(parsed)
    return _parse_single(text)


def check_equivalent(student: str, expected: str) -> Verdict:
    """Check algebraic equivalence of student response vs expected expression.

    Multi-answer inputs ('x=2, x=3') compare as sets: same size and every
    student answer matches some expected answer.
    """
    try:
        s = _safe_parse(student)
        e = _safe_parse(expected)
    except Exception as exc:
        return Verdict("unverifiable", f"Could not parse input safely: {exc}")
    try:
        s_set = list(s) if isinstance(s, tuple) else [s]
        e_set = list(e) if isinstance(e, tuple) else [e]
        if len(s_set) != len(e_set):
            if not student.strip():
                return Verdict("incomplete", "Empty response — please enter a step.")
            return Verdict("incorrect", "Not equivalent to the expected step yet.", str(s))
        unmatched = list(e_set)
        for s_item in s_set:
            hit = next((e_item for e_item in unmatched
                        if _items_equal(s_item, e_item)), None)
            if hit is None:
                return Verdict("incorrect", "Not equivalent to the expected step yet.", str(s))
            unmatched.remove(hit)
        return Verdict("correct", "Matches the expected expression.", str(s))
    except Exception as exc:
        return Verdict("unverifiable", f"Verification failed safely: {exc}")


def _items_equal(s_item, e_item) -> bool:
    with sp.evaluate(True):
        diff = sp.simplify(s_item - e_item)
    if diff == 0:
        return True
    try:
        if abs(float(s_item.evalf()) - float(e_item.evalf())) < 1e-9:
            return True
    except Exception:
        pass
    return False


def _parse_solution_set(text: str) -> list:
    """Parse 'u = 1/2 or u = 1' / '{0, pi/3}' / 'x=2, x=3' into values."""
    t = (text or "").strip().strip('"').strip("'")
    t = re.sub(r"^\{\s*(.*?)\s*\}$", r"\1", t)  # {a, b} -> a, b
    t = re.sub(r"\bor\b", ",", t, flags=re.IGNORECASE)
    parts = _split_top_level(t)
    vals = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        val = p.split("=", 1)[1] if "=" in p else p
        try:
            vals.append(_parse_single(val))
        except Exception:
            return []  # signal unparseable to caller via empty + check
    return vals


def check_solution_set(student: str, expected: str,
                       tol: float = 1e-9) -> Verdict:
    """Same solution set: order/'or'/braces insensitive, exact forms."""
    if not (student or "").strip():
        return Verdict("incomplete", "Empty response — please enter a step.")
    try:
        s_vals = _parse_solution_set(student)
        e_vals = _parse_solution_set(expected)
    except Exception as exc:
        return Verdict("unverifiable", f"Could not parse input safely: {exc}")
    if not s_vals or not e_vals:
        return Verdict("unverifiable", "Could not parse the solution set.")
    if len(s_vals) < len(e_vals):
        # Correct but incomplete: one of several solutions.
        unmatched = list(e_vals)
        for s_item in s_vals:
            hit = next((e for e in unmatched if _items_equal(s_item, e)), None)
            if hit is None:
                return Verdict("incorrect",
                               "Not equivalent to the expected step yet.")
            unmatched.remove(hit)
        return Verdict("incomplete", "You found one solution. "
                                     "Is there another?")
    if len(s_vals) != len(e_vals):
        return Verdict("incorrect", "Not equivalent to the expected step yet.")
    unmatched = list(e_vals)
    for s_item in s_vals:
        hit = next((e for e in unmatched if _items_equal(s_item, e)), None)
        if hit is None:
            return Verdict("incorrect",
                           "Not equivalent to the expected step yet.")
        unmatched.remove(hit)
    return Verdict("correct", "Matches the expected solution set.")


def check_factored(student: str, expected: str) -> Verdict:
    """Factored form: equivalent AND actually factored (not just expanded)."""
    # A trailing "= 0" on the expected answer is a display convention,
    # not part of the factored form: compare against the LHS.
    exp = expected.strip()
    if exp.endswith("= 0") or exp.endswith("=0"):
        exp = exp.rsplit("=", 1)[0]
    base = check_equivalent(student, exp)
    if base.status != "correct":
        if not (student or "").strip():
            return Verdict("incomplete", "Empty response — please enter a step.")
        return base
    # Factored means: contains a parenthesized product or power of one,
    # and is not a bare expanded polynomial.
    t = normalize_math(student if "=" not in student
                       else student.split("=", 1)[0]
                       if student.split("=", 1)[0].strip() not in ("", "0")
                       else student.split("=", 1)[1])
    has_product = bool(re.search(r"\)\s*[\*]?\s*\(|\)\s*\*\*?\s*\(?", t))
    has_paren = "(" in t
    if has_paren and (has_product or "**" in t or "^" in t):
        return Verdict("correct", "Correct factored form.", base.normalized)
    return Verdict("incorrect",
                   "That's equivalent but not factored — "
                   "write it as a product of factors.")


def check_numeric_value(student: str, expected: str,
                        tol: float = 1e-9) -> Verdict:
    """Numeric comparison within tolerance."""
    try:
        s = _parse_single(student.split("=", 1)[1]
                          if "=" in student else student)
        e = _parse_single(expected.split("=", 1)[1]
                          if "=" in expected else expected)
        sv, ev = float(s.evalf()), float(e.evalf())
    except Exception as exc:
        return Verdict("unverifiable", f"Could not parse input safely: {exc}")
    if abs(sv - ev) <= tol:
        return Verdict("correct", "Numeric value matches.")
    return Verdict("incorrect", "Not equivalent to the expected step yet.")


def check_derivative(student: str, expected: str) -> Verdict:
    """Derivative step: symbolic equality of the two expressions."""
    return check_equivalent(student, expected)


def check_integral(student: str, expected: str) -> Verdict:
    """Antiderivative step: differentiate student, compare to integrand.

    Here `expected` is the integrand and `student` the proposed
    antiderivative (+C tolerated by construction: d(+C)/dx = 0).
    """
    try:
        x = sp.Symbol("x")
        s = _parse_single(student.split("=", 1)[1]
                          if "=" in student else student)
        e = _parse_single(expected)
        diff = sp.simplify(sp.diff(s, x) - e)
        if diff == 0:
            return Verdict("correct", "Valid antiderivative (up to constant).")
        if _numeric_confirm(sp.diff(s, x), e):
            return Verdict("correct", "Valid antiderivative (up to constant).")
        return Verdict("incorrect", "Not equivalent to the expected step yet.")
    except Exception as exc:
        return Verdict("unverifiable", f"Could not parse input safely: {exc}")


def check_limit(student: str, expected: str) -> Verdict:
    """Limit step: value equality; accepts DNE/infinity wording."""
    dne = {"dne", "does not exist", "undefined", "infinity", "inf",
           "-infinity", "-inf", "oo", "-oo"}
    s_low = (student or "").strip().lower()
    e_low = (expected or "").strip().lower()
    if s_low in dne or e_low in dne:
        if s_low in dne and e_low in dne:
            # Both non-finite: match signs loosely (inf vs oo).
            neg_s = s_low.startswith("-")
            neg_e = e_low.startswith("-")
            if neg_s == neg_e:
                return Verdict("correct", "Matches the expected limit.")
            return Verdict("incorrect",
                           "Not equivalent to the expected step yet.")
        return Verdict("incorrect", "Not equivalent to the expected step yet.")
    return check_equivalent(student, expected)


def check_step_typed(expected: str, student: str,
                     check_type: str = "equivalent_expression") -> Verdict:
    """Dispatch one guided-step check by check_type (software decides)."""
    ct = (check_type or "equivalent_expression").strip().lower()
    if ct == "solution_set":
        return check_solution_set(student, expected)
    if ct == "factored_form":
        return check_factored(student, expected)
    if ct == "numeric_value":
        return check_numeric_value(student, expected)
    if ct == "derivative":
        return check_derivative(student, expected)
    if ct == "integral":
        return check_integral(student, expected)
    if ct == "limit":
        return check_limit(student, expected)
    if ct in ("identity_step", "equivalent_expression",
              "equivalent_equation"):
        return check_equivalent(student, expected)
    # Unknown types incl. "conceptual": caller handles LLM two-pass.
    return check_equivalent(student, expected)


def display_math(text: str) -> str:
    """Small pretty-printer for tutor replies (common cases only).

    Converts CAS spellings to clean display: '2*x - 1' -> '2x - 1',
    'x**2' -> 'x^2', 'cos(x)' stays, 'pi' -> 'π'. Not a full
    CAS-to-display renderer; unknown shapes pass through unchanged.
    """
    t = "" if text is None else str(text)
    # Drop explicit * in coefficient/variable juxtaposition: 2*x -> 2x.
    t = re.sub(r"(?<![a-zA-Z0-9_.])(\d+)\s*\*\s*(?=[a-zA-Z(pi])", r"\1", t)
    t = re.sub(r"\*\s*(?=[a-zA-Z(])", "", t)
    # ** power -> ^ power.
    t = re.sub(r"\*\*(\d+)", r"^\1", t)
    t = re.sub(r"\*\*\(([^()]*)\)", r"^(\1)", t)
    # pi symbol (word-boundary safe).
    t = re.sub(r"(?<![a-zA-Z])pi(?![a-zA-Z])", "π", t)
    t = re.sub(r"(?<![a-zA-Z])theta(?![a-zA-Z])", "θ", t)
    t = re.sub(r"(?<![a-zA-Z])oo(?![a-zA-Z])", "∞", t)
    t = re.sub(r"\bsqrt\(([^()]*)\)", r"√(\1)", t)
    return t.strip()


def check_numeric(student: str, expected: float, tol: float = 1e-6) -> Verdict:
    try:
        s = _safe_parse(student)
        val = float(s.evalf())
    except Exception as exc:
        return Verdict("unverifiable", f"Could not parse input safely: {exc}")
    if abs(val - expected) < tol:
        return Verdict("correct", "Numeric value matches.", str(val))
    return Verdict("incorrect", f"Expected approximately {expected}.", str(val))


# --- Canonical solver (Phase 3): SymPy independently computes the answer ---
#
# Routing is explicit via objective_tags against MODE_TAGS, never guessed
# from punctuation. Unknown modes fall back to equivalence checking.
# Every handler returns Verdict (correct|incorrect|unverifiable).

MODE_TAGS: dict[str, str] = {
    # equations
    "linear equation": "equation", "linear equations": "equation",
    "quadratic": "equation", "solve equation": "equation",
    "solving": "equation", "extraneous roots": "equation",
    "rational equation": "equation", "rational equations": "equation",
    "system": "system", "linear system": "system",
    "systems of equations": "system", "system of linear equations": "system",
    # calculus
    "derivative": "deriv", "differentiation": "deriv",
    "differentiate": "deriv", "chain rule": "deriv",
    "rate of change": "deriv", "related rates": "deriv",
    "integral": "integ", "integration": "integ",
    "antiderivative": "integ", "area under curve": "integ",
    "limit": "limit", "limits": "limit", "continuity": "limit",
    # simplification
    "simplify": "simplify", "simplification": "simplify",
    "identity": "simplify", "prove identity": "simplify",
    "factor": "simplify", "factoring": "simplify",
    "expand": "simplify",
    # functions & transcendental (verified via equivalence fallback)
    "power function": "simplify", "power functions": "simplify",
    "exponential": "simplify", "logarithm": "simplify",
    "logarithms": "simplify",
}

# Fixed probe points for numeric confirmation (deterministic, no randomness).
PROBE_POINTS = (1, 2, -1)

# Constrained grammar for limit specs the generator may emit:
#   \lim_{x \to 2} <expr>   or   lim_{x->2} <expr>   or   limit x->2: <expr>
# The point may be a number, \infty / -\infty, or an expression.
# Braces are balanced-aware: the point group cannot contain an
# unbalanced '}' and the expression runs to end of string.
_LIMIT_RE = re.compile(
    r"(?:\\lim|\\operatorname\{lim\}|lim(?:it)?)\s*[_]?\{?\s*"
    r"([a-zA-Z])\s*(?:\\to|->)\s*"
    r"(\\infty\s*|-\s*\\infty\s*|[^\\{}]+?)\s*\}\s*[:;,]?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _strip_latex_environments(text: str) -> str:
    """Unwrap display-math and alignment environments into plain equations.

    Handles \\begin{cases}...\\end{cases} / aligned / matrix (row separator
    \\\\ -> comma) and $$ \\[ \\] delimiters, so system prompts parse.
    """
    t = text.strip()
    # \\begin{env} ... \\end{env} -> inner content, with \\\\ rows -> commas.
    # Single-backslash form (actual model output) AND double-backslash form
    # (JSON-escaped round-trips) are both handled.
    def _unwrap(m):
        inner = m.group(2)
        inner = re.sub(r"\\\\", ",", inner)
        inner = re.sub(r"&", " ", inner)
        return " " + inner + " "
    t = re.sub(r"\\{1,2}begin\{([a-zA-Z*]+)\}(.*?)\\{1,2}end\{\1\}",
               _unwrap, t, flags=re.DOTALL)
    # Display delimiters -> spaces (content already plain after unwrap).
    t = t.replace("$$", " ")
    t = re.sub(r"\\\[|\\\]", " ", t)
    return t.strip()


def detect_mode(question) -> str:
    """Explicit routing from step objective_tags. Falls back to 'equiv'.

    A limit written in the prompt ("\\lim_{x \\to 1} x") is untagged but has a
    canonical solver, and the equivalence fallback would grade its correct
    value as wrong, so the prompt itself outranks the generic fallback.
    """
    seen: list[str] = []
    for step in question.steps:
        for tag in step.objective_tags:
            mode = MODE_TAGS.get(tag.strip().lower())
            if mode and mode not in seen:
                seen.append(mode)
    if not seen:
        # "\b" cannot end \lim here: the usual form is \lim_{...}, and "_" is
        # a word character, so the guard is "no letter follows".
        if re.search(r"\\lim(?![a-zA-Z])",
                     str(getattr(question, "prompt_latex", "") or "")):
            return "limit"
        return "equiv"
    # Prefer the most specific handler when tags mix.
    for preferred in ("system", "equation", "deriv", "integ",
                      "limit", "simplify"):
        if preferred in seen:
            return preferred
    return "equiv"


def _eq_objects(prompt: str) -> tuple[list, list]:
    """Parse 'LHS = RHS [, ...]' prompt into ([Eq...], [symbols]).

    Symbols are collected from the raw sides BEFORE subtraction, so
    cancelling equations like 'x=x+1' still report their variables.
    Leading English prose ('Solve ...', 'Find ...') is stripped: only
    segments containing '=' or math operators count as equations.
    """
    import string as _string

    parts = _split_top_level(_strip_latex_environments(prompt))
    eqs, syms = [], []
    # Merge newline-split rows: '...equations:\n2x+y=7,\nx-y=2' splits on
    # commas AND newlines must not orphan the prose lead-in.
    flat: list[str] = []
    for part in parts:
        flat.extend(re.split(r"\n+", part))
    parts = [p.strip() for p in flat if p.strip()]
    for i, part in enumerate(parts):
        # First segment may carry the prose lead-in ('Solve ... :').
        if i == 0 and "=" in part:
            part = re.sub(r"^.*?:\s*(?=[a-zA-Z0-9\\$(\[][^=]*=)",
                          "", part).strip()
        # Cut English lead-in: 'Solve x+y=3' -> 'x+y=3'. Only strips when
        # the prefix contains no '=' (so 'x=2, y=1' answers are untouched).
        if "=" in part:
            lead = re.match(
                r"^([A-Za-z][A-Za-z\s',]*?)\s+"
                r"(?=[a-zA-Z0-9\\$(\[][^=]*=)", part)
            if lead and "=" not in lead.group(1):
                part = part[len(lead.group(1)):].strip()
        if "=" not in part and not re.search(r"[0-9\\^+\-*/()]", part):
            continue  # pure prose segment, not an equation
        if "=" not in part:
            # Bare expression: treat as expr = 0.
            expr = _parse_single(part)
            syms.extend(list(expr.free_symbols))
        else:
            lhs_raw, rhs_raw = part.split("=", 1)
            lhs = _parse_single(lhs_raw) if lhs_raw.strip() else sp.Integer(0)
            rhs = _parse_single(rhs_raw) if rhs_raw.strip() else sp.Integer(0)
            syms.extend(list(lhs.free_symbols))
            syms.extend(list(rhs.free_symbols))
            expr = lhs - rhs
        eqs.append(sp.Eq(expr, 0))
    # Stable symbol order: x, y, z… then the rest alphabetically.
    order = {name: i for i, name in enumerate(
        ["x", "y", "z", "t", "w", "u", "v"])}
    ordered = sorted({str(s): s for s in syms}.values(),
                     key=lambda s: (order.get(str(s), 99), str(s)))
    return eqs, ordered


def _parse_answer_set(answer: str) -> tuple[list, str | None]:
    """Parse LLM answer into a list of assignment dicts or expressions.

    Returns (items, error). For 'x=2, y=3' items are {x: 2, y: 3};
    for bare 'x=2, x=3' items are [2, 3] with implied single variable.
    """
    try:
        parts = _split_top_level(
            answer.split("=", 1)[1] if answer.count("=") == 0 else answer)
        if answer.count("=") == 0:
            return [_parse_single(p) for p in parts], None
        assigns: dict[str, list] = {}
        for part in parts:
            if "=" not in part:
                return [], f"mixed answer format: {part!r}"
            var, val = part.split("=", 1)
            var = normalize_math(var).strip()
            if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", var):
                return [], f"bad variable name: {var!r}"
            assigns.setdefault(var, []).append(_parse_single(val))
        # Single variable with several values -> value list.
        if len(assigns) == 1:
            return list(next(iter(assigns.values()))), None
        # Multi variable: each must have exactly one value -> solution tuple.
        if any(len(v) != 1 for v in assigns.values()):
            return [], "multi-variable answer needs one value each"
        ordered = sorted(assigns)
        return [{k: assigns[k][0] for k in ordered}], None
    except Exception as exc:
        return [], str(exc)


def _solutions_to_set(solutions, syms: list) -> list | None:
    """Normalize sympy solve/linsolve output to a comparable solution list."""
    if not solutions:
        return []
    if isinstance(solutions, sp.FiniteSet):
        out = []
        for s in solutions:
            out.append(tuple(s) if isinstance(s, (tuple, sp.Tuple)) else s)
        return sorted(out, key=str)
    if isinstance(solutions, tuple) and len(solutions) == 2 and isinstance(
            solutions[1], sp.Matrix):
        # linsolve: (symbols, Matrix) — rows are solution tuples.
        mat = solutions[1]
        return [tuple(row) for row in mat.tolist()]
    rows = list(solutions)
    if rows and isinstance(rows[0], (tuple, list, sp.Tuple)):
        return [tuple(r) for r in rows]
    if len(syms) == 1:
        return rows
    return [(r,) for r in rows]


def _verify_equation(question, is_system: bool) -> Verdict:
    """Canonical equation check: solve prompt with SymPy, compare sets."""
    try:
        eqs, syms = _eq_objects(question.prompt_latex)
    except Exception as exc:
        return Verdict("unverifiable", f"Could not parse problem: {exc}")
    if not syms:
        return Verdict("unverifiable", "No variables found in problem.")
    try:
        if is_system or len(eqs) > 1:
            try:
                raw = sp.linsolve(eqs, syms)
                expected = _solutions_to_set(raw, syms)
            except Exception:
                raw = sp.nonlinsolve(eqs, syms)
                expected = _solutions_to_set(raw, syms)
        else:
            expected = sp.solve(eqs[0], syms, dict=False)
            expected = _solutions_to_set(expected, syms)
    except Exception as exc:
        return Verdict("unverifiable", f"Solver failed safely: {exc}")
    items, err = _parse_answer_set(question.final_answer_latex)
    if err is not None:
        return Verdict("unverifiable", f"Could not parse answer: {err}")
    if expected is None:
        return Verdict("unverifiable", "Solver returned no usable result.")
    if not expected:
        # Canonical solution set is empty: no-solution (or 0=0 identity).
        # Any claimed solution is incorrect; only an explicit "no solution"
        # style answer could pass, which the answer grammar cannot express.
        if not question.final_answer_latex.strip():
            return Verdict("incomplete",
                           "Empty response — please enter a step.")
        return Verdict("incorrect",
                       "The canonical solution set is empty; "
                       "no claimed solution can be correct.")
    try:
        if len(items) != len(expected):
            if not question.final_answer_latex.strip():
                return Verdict("incomplete",
                               "Empty response — please enter a step.")
            return Verdict("incorrect",
                           f"Expected {len(expected)} solution(s), "
                           f"got {len(items)}.")
        unmatched = list(expected)
        for item in items:
            hit = None
            for cand in unmatched:
                if _solution_matches(item, cand, syms):
                    hit = cand
                    break
            if hit is None:
                return Verdict("incorrect",
                               "Solution set does not match the "
                               "canonical solution.")
            unmatched.remove(hit)
        return Verdict("correct", "Matches the canonical solution set.")
    except Exception as exc:
        return Verdict("unverifiable", f"Comparison failed safely: {exc}")


def _solution_matches(item, cand, syms: list) -> bool:
    """One LLM solution vs one canonical solution (symbolic, then numeric)."""
    if isinstance(item, dict):
        # Multi-variable assignment: compare per variable.
        if not isinstance(cand, (tuple, list)) or len(cand) != len(syms):
            return False
        lookup = {str(s): v for s, v in zip(syms, cand)}
        names = sorted(item)
        if [str(s) for s in syms] != names and set(names) != set(
                str(s) for s in syms):
            return False
        try:
            ordered_cand = tuple(lookup[k] for k in names)
        except KeyError:
            return False
        return all(_items_equal(sp.sympify(v), c)
                   for v, c in zip(
                       [item[k] for k in names], ordered_cand))
    values = item if isinstance(item, (tuple, list)) else (item,)
    cands = cand if isinstance(cand, (tuple, list)) else (cand,)
    if len(values) != len(cands):
        return False
    return all(_items_equal(sp.sympify(v), c)
               for v, c in zip(values, cands))


def _parse_function_of_x(text: str):
    """Parse prompt/answer text as f(x): strips 'f(x) =' / 'y =' prefixes."""
    t = text.strip()
    t = re.sub(r"^[a-zA-Z]\s*\([a-zA-Z]\)\s*=", "", t).strip()
    t = re.sub(r"^[a-zA-Z]\s*=", "", t).strip()
    expr = _parse_single(t)
    if isinstance(expr, tuple):
        raise ValueError("expected a single expression")
    return expr


def _verify_calculus(question, mode: str) -> Verdict:
    """Canonical calculus check: differentiate/integrate/limit via SymPy."""
    x = sp.Symbol("x")
    try:
        if mode == "limit":
            m = _LIMIT_RE.search(question.prompt_latex.strip())
            if not m:
                return Verdict(
                    "unverifiable",
                    "Limit spec not in supported form "
                    "(e.g. \\lim_{x \\to 2} x^2).")
            var = sp.Symbol(m.group(1))
            point_raw = m.group(2).strip()
            # \infty / -\infty (single or JSON-double-escaped backslashes).
            if re.fullmatch(r"\\{1,2}infty", point_raw):
                point = sp.oo
            elif re.fullmatch(r"-\s*\\{1,2}infty", point_raw):
                point = -sp.oo
            else:
                try:
                    point = _parse_single(point_raw)
                except Exception:
                    return Verdict(
                        "unverifiable",
                        f"Could not parse limit point: {m.group(2)!r}")
            expr = _parse_single(m.group(3))
            expected = sp.limit(expr, var, point)
            answer = _parse_single(question.final_answer_latex)
            return (Verdict("correct", "Matches the canonical limit.")
                    if _items_equal(answer, expected)
                    else Verdict("incorrect",
                                 "Does not match the canonical limit."))
        f_expr = _parse_function_of_x(question.prompt_latex)
        g_expr = _parse_function_of_x(question.final_answer_latex)
        if mode == "deriv":
            expected = sp.diff(f_expr, x)
            return (Verdict("correct", "Matches the canonical derivative.")
                    if _items_equal(g_expr, expected)
                    else Verdict("incorrect",
                                 "Does not match the canonical derivative."))
        if mode == "integ":
            # Ignore +C: differentiate the ANSWER and compare to integrand.
            diff = sp.simplify(sp.diff(g_expr, x) - f_expr)
            if diff == 0:
                return Verdict("correct",
                               "Valid antiderivative (up to constant).")
            return Verdict("incorrect",
                           "Derivative of answer does not match integrand.")
    except Exception as exc:
        return Verdict("unverifiable", f"Could not parse input safely: {exc}")
    return Verdict("unverifiable", f"Unknown calculus mode: {mode}")


def _numeric_confirm(a, b, points=PROBE_POINTS, required: int = 3) -> bool:
    """Fixed-point numeric equality: all required probes agree, skips holes."""
    hits = 0
    for pt in points:
        try:
            va = complex(a.subs("x", pt).evalf())
            vb = complex(b.subs("x", pt).evalf())
        except Exception:
            continue
        if abs(va - vb) < 1e-6:
            hits += 1
    return hits >= required


def _verify_simplify(question) -> Verdict:
    """Equivalence check: symbolic first, fixed-point numeric confirmation."""
    try:
        s = _parse_single(question.prompt_latex)
        e = _parse_single(question.final_answer_latex)
    except Exception as exc:
        return Verdict("unverifiable", f"Could not parse input safely: {exc}")
    try:
        with sp.evaluate(True):
            diff = sp.simplify(s - e)
        if diff == 0:
            return Verdict("correct", "Expressions are equivalent.")
        if _numeric_confirm(s, e):
            return Verdict("correct",
                           "Numerically equivalent at probe points.")
        if not question.final_answer_latex.strip():
            return Verdict("incomplete", "Empty response — please enter a step.")
        return Verdict("incorrect", "Not equivalent to the expected form.")
    except Exception as exc:
        return Verdict("unverifiable", f"Verification failed safely: {exc}")


def verify_canonical(question) -> Verdict:
    """Canonical-solver dispatcher: SymPy computes, LLM output is compared.

    Routing comes from explicit objective_tags (detect_mode); unknown or
    untagged questions fall back to algebraic equivalence. Returns Verdict.
    """
    mode = detect_mode(question)
    if mode in ("equation", "system"):
        return _verify_equation(question, is_system=(mode == "system"))
    if mode in ("deriv", "integ", "limit"):
        return _verify_calculus(question, mode)
    if mode == "simplify":
        return _verify_simplify(question)
    # Default for unknown/untagged topics: try the equation solver first
    # (handles bare 'x+1=2' prompts with no tags), then fall back to the
    # simplify check (trig identities, logs, expressions). Never crashes.
    try:
        eq_verdict = _verify_equation(question, is_system=False)
        if eq_verdict.status in ("correct", "incorrect"):
            return eq_verdict
    except Exception:
        pass
    try:
        return _verify_simplify(question)
    except Exception:
        pass
    # Default: algebraic equivalence of final answer vs last step.
    if question.final_answer_latex and question.steps:
        return check_equivalent(question.final_answer_latex,
                                question.steps[-1].expected_input_latex)
    return Verdict("unverifiable", "Nothing to verify.")


def canonical_answer(problem: str) -> list[str]:
    """SymPy's own answer(s) to a problem, for reply leak checks.

    Only the shapes the canonical solvers cover: a limit's value and an
    equation's solution set. Empty when neither applies — a bare expression
    ("3x-9") has no single answer, and guessing one would censor correct
    guidance.
    """
    text = (problem or "").strip()
    if not text:
        return []
    try:
        m = _LIMIT_RE.search(text)
        if m:
            point_raw = m.group(2).strip()
            if re.fullmatch(r"\\{1,2}infty", point_raw):
                point = sp.oo
            elif re.fullmatch(r"-\s*\\{1,2}infty", point_raw):
                point = -sp.oo
            else:
                point = _parse_single(point_raw)
            val = sp.simplify(
                sp.limit(_parse_single(m.group(3)), sp.Symbol(m.group(1)),
                        point))
            return [display_math(val)]
    except Exception:
        return []
    if "=" not in text:
        return []
    try:
        eqs, syms = _eq_objects(text)
        if not eqs or not syms or len(eqs) > 1:
            return []
        return [display_math(sp.simplify(sol))
                for sol in sp.solve(eqs[0], syms, dict=False)]
    except Exception:
        return []
