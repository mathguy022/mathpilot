"""Floating-tutor linear-equation solver: CAS-backed step plans.

Deterministic SymPy logic. Generates a hidden solution_plan (ordered list
of {instruction, expected_expression, reason}) for one-variable linear
equations, and verifies student steps by algebraic equivalence — so
"9 = 3x", "3x = 9", and "3x = 10 - 1" all pass Step 1 of "10 = 3x + 1".

Non-linear / unparseable input returns None so the caller can use the
safe fallback (ask the student to explain their reasoning).
"""
from __future__ import annotations

import re

import sympy as sp

from .verifier import _parse_single, check_equivalent


def _symbols(expr) -> list:
    try:
        return sorted(expr.free_symbols, key=str)
    except Exception:
        return []


def build_linear_plan(problem_text: str) -> list[dict] | None:
    """Build a step plan for a one-variable linear equation.

    Returns None when the input is not a solvable one-variable linear
    equation (caller falls back to generic Socratic guidance).
    """
    text = (problem_text or "").strip().strip('"').strip("'")
    if "=" not in text:
        return None
    try:
        lhs = _parse_single(text.split("=", 1)[0])
        rhs = _parse_single(text.split("=", 1)[1])
    except Exception:
        return None
    if isinstance(lhs, tuple) or isinstance(rhs, tuple):
        return None
    syms = sorted(set(_symbols(lhs)) | set(_symbols(rhs)), key=str)
    if len(syms) != 1:
        return None
    x = syms[0]
    try:
        eq = sp.Eq(lhs - rhs, 0)
        sol = sp.solve(eq, x, dict=False)
    except Exception:
        return None
    if not sol:
        return None
    solution = sol[0]
    # Must be linear in x: degree check on the cleared polynomial.
    try:
        poly = sp.Poly(lhs - rhs, x)
        if poly.degree() != 1:
            return None
    except Exception:
        return None

    def s(e) -> str:
        try:
            return str(sp.nsimplify(sp.simplify(e), rational=True))
        except Exception:
            return str(sp.simplify(e))

    steps: list[dict] = []
    diff = sp.expand(lhs - rhs)
    const_term = complex(diff.subs(x, 0).evalf())
    coeff = complex(sp.expand(lhs - rhs).coeff(x).evalf())
    lhs_has_x = bool(lhs.has(x))
    rhs_has_x = bool(rhs.has(x))

    def mag_of(c) -> str:
        # Rational-preserving display: pi/4, not 98174770424681/12500000...
        try:
            v = sp.nsimplify(abs(complex(c).real), rational=False)
            if v == int(v):
                return str(int(v))
            return str(v)
        except Exception:
            pass
        v = float(abs(complex(c).real))
        return str(int(v)) if v == int(v) else s(abs(complex(c)))

    if lhs_has_x != rhs_has_x:
        # x on one side only: undo the constant on the x side first,
        # keeping the original side order (9 = 3x, not -3x = -9).
        x_expr = lhs if lhs_has_x else rhs
        other_expr = rhs if lhs_has_x else lhs
        const_side = complex(sp.expand(x_expr).subs(x, 0).evalf())
        if abs(const_side) > 1e-9:
            op = "Subtract" if const_side.real > 0 else "Add"
            mag = mag_of(const_side)
            new_x = sp.expand(x_expr - const_side)
            new_other = sp.expand(other_expr - const_side)
            if lhs_has_x:
                exp = f"{s(new_x)} = {s(new_other)}"
            else:
                exp = f"{s(new_other)} = {s(new_x)}"
            prep = "from" if op == "Subtract" else "to"
            steps.append({
                "instruction": f"{op} {mag} {prep} both sides.",
                "expected_expression": exp,
                "reason": "Whatever you do to one side, do to the other.",
            })
            work_lhs = new_x if lhs_has_x else new_other
            work_rhs = new_other if lhs_has_x else new_x
        else:
            work_lhs, work_rhs = lhs, rhs
    else:
        # x on both sides: collect to coeff*x = -const as step 1.
        if abs(const_term) > 1e-9:
            op = "Subtract" if const_term.real > 0 else "Add"
            prep = "from" if op == "Subtract" else "to"
            steps.append({
                "instruction": f"{op} {mag_of(const_term)} {prep} both sides.",
                "expected_expression": f"{s(coeff * x)} = {s(-const_term)}",
                "reason": "Get all the x terms on one side first.",
            })
        work_lhs, work_rhs = coeff * x, sp.sympify(-const_term)
        lhs_has_x = True

    # Coefficient of x on the x side after step 1.
    try:
        side_expr = work_lhs if lhs_has_x else work_rhs
        coeff_side = complex(sp.expand(side_expr).coeff(x).evalf())
    except Exception:
        coeff_side = coeff
    if abs(abs(coeff_side) - 1.0) > 1e-9 and abs(coeff_side) > 1e-9:
        try:
            # Symbolic display: pi/4 and 1/2, never 0.785... or 0.500000000000000.
            side_sym = sp.expand(work_lhs if lhs_has_x else work_rhs)
            coef_sym = sp.simplify(side_sym.coeff(x))
            disp = s(coef_sym) if coef_sym != 0 else s(coeff_side)
        except Exception:
            disp = s(coeff_side)
        steps.append({
            "instruction": f"Divide both sides by {disp}.",
            "expected_expression": f"{x} = {s(solution)}",
            "reason": "Isolate x by undoing the multiplication.",
        })
    elif not steps:
        steps.append({
            "instruction": "Get x alone on one side.",
            "expected_expression": f"{x} = {s(solution)}",
            "reason": "Isolate the variable.",
        })

    # The check step substitutes into the side that originally held x.
    x_expr_side = lhs if lhs_has_x else rhs
    check_side = rhs if lhs_has_x else lhs
    try:
        check_value = s(check_side.subs(x, solution) if check_side.has(x)
                        else check_side)
    except Exception:
        check_value = s(solution)
    steps.append({
        "instruction": f"Check: put {x} = {s(solution)} into {s(x_expr_side)}.",
        "expected_expression": s(check_value),
        "reason": "Substituting back confirms the answer.",
    })
    return steps


def build_quadratic_plan(problem_text: str) -> list[dict] | None:
    """Step plan for a one-variable quadratic equation.

    Returns None when the input is not a solvable quadratic. Instructions
    are actions, never results; expected forms are CAS-parsable and each
    carries a check_type for the typed verifier.
    """
    from .verifier import normalize_math

    text = (problem_text or "").strip().strip('"').strip("'")
    if "=" not in text:
        return None
    try:
        lhs = _parse_single(text.split("=", 1)[0])
        rhs = _parse_single(text.split("=", 1)[1])
    except Exception:
        return None
    if isinstance(lhs, tuple) or isinstance(rhs, tuple):
        return None
    syms = sorted(set(_symbols(lhs)) | set(_symbols(rhs)), key=str)
    if len(syms) != 1:
        return None
    x = syms[0]
    try:
        poly = sp.Poly(sp.expand(lhs - rhs), x)
        if poly.degree() != 2:
            return None
    except Exception:
        return None
    try:
        sols = sp.solve(sp.Eq(lhs, rhs), x, dict=False)
    except Exception:
        return None
    if not sols:
        return None

    def s(e) -> str:
        try:
            return str(sp.nsimplify(sp.simplify(e), rational=True))
        except Exception:
            return str(sp.simplify(e))

    steps: list[dict] = []
    std = sp.expand(lhs - rhs)
    # Step 1: standard form (skip if already ax^2+bx+c = 0).
    try:
        rhs_zero = sp.simplify(rhs) == 0
    except Exception:
        rhs_zero = False
    if not rhs_zero:
        steps.append({
            "instruction": "Move everything to one side to get zero on the other side.",
            "expected_expression": f"{s(std)} = 0",
            "check_type": "equivalent_equation",
            "reason": "Standard form lets us factor or use the formula.",
            "micro_steps": ["Subtract the right side from both sides."],
        })
    # Step 2: factor (if it factors over rationals), else quadratic formula.
    try:
        factored = sp.factor(std)
    except Exception:
        factored = None
    factors_cleanly = (factored is not None and factored != std
                       and factored != -std)
    if factors_cleanly:
        steps.append({
            "instruction": ("Find two numbers that multiply to the constant term "
                            "and add to the middle coefficient, then write the factored form."),
            "expected_expression": f"{s(factored)} = 0",
            "check_type": "factored_form",
            "reason": "Factoring splits the quadratic into linear pieces.",
            "micro_steps": ["List factor pairs of the constant term.",
                            "Pick the pair that adds to the middle coefficient."],
        })
        roots = sorted(sols, key=str)
        steps.append({
            "instruction": "Set each factor equal to zero and solve.",
            "expected_expression": " or ".join(f"{x} = {s(r)}" for r in roots),
            "check_type": "solution_set",
            "reason": "A product is zero when some factor is zero.",
            "micro_steps": ["Solve the first factor.", "Solve the second factor."],
        })
    else:
        a, b, c = poly.nth(2), poly.nth(1), poly.nth(0)
        steps.append({
            "instruction": ("The quadratic formula gives the solutions directly — "
                            "write it with your a, b, and c filled in."),
            "expected_expression": " or ".join(f"{x} = {s(r)}" for r in sorted(sols, key=str)),
            "check_type": "solution_set",
            "reason": "The formula works even when factoring does not.",
            "micro_steps": ["Identify a, b, c.", "Compute the discriminant.",
                            "Simplify the two values."],
        })
    steps.append({
        "instruction": f"Check one solution in the original equation.",
        "expected_expression": s(std.subs(x, sols[0])),
        "check_type": "numeric_value",
        "reason": "Substituting back confirms the answer.",
        "micro_steps": [],
    })
    return steps


def build_trig_quad_plan(problem_text: str) -> list[dict] | None:
    """Step plan for trig equations that reduce to a quadratic in one func.

    Example: 2cos^2 x - 3cos x + 1 = 0 with solutions in [0, 2*pi).
    Without an explicit interval the plan's first step is a clarifying
    question (caller renders it and stores the answer before proceeding).
    """
    from .verifier import normalize_math

    text = (problem_text or "").strip().strip('"').strip("'")
    # Strip a trailing interval clause before parsing:
    # "... = 0 in [0, 2*pi)" -> equation "... = 0", interval kept aside.
    interval_note = ""
    m = re.search(r"\bin\s*([\[\(][^\]\)]*[\]\)])\s*$", text)
    if m:
        interval_note = "in " + m.group(1)
        text = text[:m.start()].strip()
    if "=" not in text:
        return None
    try:
        lhs = _parse_single(text.split("=", 1)[0])
        rhs = _parse_single(text.split("=", 1)[1])
    except Exception:
        return None
    if isinstance(lhs, tuple) or isinstance(rhs, tuple):
        return None
    expr = sp.expand(lhs - rhs)
    # Find exactly one trig function of one argument.
    funcs = [f for f in expr.atoms(sp.Function)
             if str(f.func) in ("sin", "cos", "tan")]
    if not funcs:
        return None
    names = {str(f.func) for f in funcs}
    args = {str(f.args[0]) for f in funcs if f.args}
    if len(names) != 1 or len(args) != 1:
        return None
    fname, arg = names.pop(), args.pop()
    try:
        u = sp.Symbol("u")
        fexpr = getattr(sp, fname)(sp.Symbol(arg))
        subbed = sp.expand(expr.subs(fexpr, u))
        poly = sp.Poly(subbed, u)
        if poly.degree() != 2:
            return None
    except Exception:
        return None
    try:
        u_sols = sp.solve(sp.Eq(subbed, 0), u, dict=False)
    except Exception:
        return None
    if not u_sols:
        return None

    def s(e) -> str:
        try:
            return str(sp.nsimplify(sp.simplify(e), rational=True))
        except Exception:
            return str(sp.simplify(e))

    has_interval = bool(interval_note)
    steps: list[dict] = []
    if not has_interval:
        steps.append({
            "instruction": ("Should we find all solutions, or only those "
                            "in [0, 2*pi)?"),
            "expected_expression": "[0, 2*pi)",
            "check_type": "conceptual",
            "reason": "Trig equations need an interval to list solutions.",
            "micro_steps": [],
        })
        interval_note = "in [0, 2*pi)"
    steps.append({
        "instruction": f"Let u = {fname}({arg}). Rewrite the equation using u.",
        "expected_expression": f"{s(subbed)} = 0",
        "check_type": "equivalent_equation",
        "reason": "Substitution turns the trig equation into a quadratic.",
        "micro_steps": [f"Replace every {fname}({arg}) with u."],
    })
    try:
        factored = sp.factor(subbed)
    except Exception:
        factored = None
    if factored is not None and factored != subbed and factored != -subbed:
        steps.append({
            "instruction": "Factor the quadratic in u.",
            "expected_expression": f"{s(factored)} = 0",
            "check_type": "factored_form",
            "reason": "Factoring splits the quadratic into linear pieces.",
            "micro_steps": [],
        })
    steps.append({
        "instruction": "Set each factor to zero and solve for u.",
        "expected_expression": " or ".join(f"u = {s(r)}" for r in sorted(u_sols, key=str)),
        "check_type": "solution_set",
        "reason": "A product is zero when some factor is zero.",
        "micro_steps": [],
    })
    steps.append({
        "instruction": f"Replace u with {fname}({arg}) in each answer.",
        "expected_expression": " or ".join(
            f"{fname}({arg}) = {s(r)}" for r in sorted(u_sols, key=str)),
        "check_type": "equivalent_equation",
        "reason": "Back-substitution returns to the original variable.",
        "micro_steps": [],
    })
    # Solve each back-substituted equation on [0, 2*pi).
    x = sp.Symbol(arg)
    all_sols = []
    for r in u_sols:
        try:
            xs = sp.solve(sp.Eq(getattr(sp, fname)(x), r),
                          x, dict=False)
            for v in xs:
                nv = complex(v.evalf())
                if abs(nv.imag) < 1e-9 and -1e-9 <= nv.real < 2 * float(sp.pi) - 1e-9:
                    all_sols.append(v)
        except Exception:
            continue
    sol_str = "{" + ", ".join(s(v) for v in sorted(set(all_sols), key=str)) + "}" if all_sols else ""
    steps.append({
        "instruction": f"Find all x {interval_note} for each equation.",
        "expected_expression": sol_str,
        "check_type": "solution_set",
        "reason": "Each trig value repeats; list every angle in the interval.",
        "micro_steps": ["Use the unit circle for each value."],
    })
    steps.append({
        "instruction": "Check one solution in the original equation.",
        "expected_expression": "0 = 0",
        "check_type": "equivalent_equation",
        "reason": "Substituting back confirms the answer.",
        "micro_steps": [],
    })
    return steps


def build_limit_plan(problem_text: str) -> list[dict] | None:
    """Step plan for a limit problem (Example C shape).

    Accepts 'limit as x->2 of (x^2-4)/(x-2)', 'lim_{x->2} ...', etc.
    """
    text = (problem_text or "").strip().strip('"').strip("'")
    low = text.lower()
    if "lim" not in low and "limit" not in low:
        return None
    # Extract: variable, point, expression.
    m = re.search(r"(?:lim(?:it)?\s*(?:as)?|lim_?)\s*"
                  r"(?:\{?\s*)?([a-zA-Z])\s*(?:->|to)\s*"
                  r"([-\d.a-zA-Z+*/^() ]+?)\s*(?:of|:|,|\})?\s*(.+)$",
                  text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    var, point_raw, expr_raw = m.group(1), m.group(2).strip(), m.group(3).strip()
    expr_raw = re.sub(r"^[\s:,ofOF]+\s*", "", expr_raw)
    try:
        from .verifier import _parse_single
        x = sp.Symbol(var)
        point = _parse_single(point_raw)
        expr = _parse_single(expr_raw)
        expected = sp.limit(expr, x, point)
    except Exception:
        return None

    def s(e) -> str:
        try:
            return str(sp.nsimplify(sp.simplify(e), rational=True))
        except Exception:
            return str(sp.simplify(e))

    try:
        direct = expr.subs(x, point)
        if direct in (sp.nan, sp.zoo, sp.oo, -sp.oo):
            is_zero_over_zero = True
        else:
            num0 = sp.simplify(sp.together(direct).as_numer_denom()[0])
            den0 = sp.simplify(sp.together(direct).as_numer_denom()[1])
            is_zero_over_zero = num0 == 0 and den0 == 0
    except Exception:
        is_zero_over_zero = False
    steps = [{
        "instruction": "Try direct substitution. What form do you get?",
        "expected_expression": "0/0" if is_zero_over_zero else s(direct),
        "check_type": "equivalent_expression",
        "reason": "Substitution tells us whether the limit needs algebra.",
        "micro_steps": [f"Replace {var} with {s(point)}."],
    }]
    if is_zero_over_zero:
        try:
            factored = sp.factor(expr)
        except Exception:
            factored = None
        if factored is not None and factored != expr:
            steps.append({
                "instruction": "Factor the numerator.",
                "expected_expression": s(factored),
                "check_type": "equivalent_expression",
                "reason": "Factoring may reveal a cancelable common factor.",
                "micro_steps": ["Look for a difference of squares or common factor."],
            })
            try:
                cancelled = sp.simplify(factored)
            except Exception:
                cancelled = None
            if cancelled is not None and cancelled != factored:
                steps.append({
                    "instruction": "Cancel the common factor, then evaluate the limit.",
                    "expected_expression": s(expected),
                    "check_type": "limit",
                    "reason": "After canceling, direct substitution works.",
                    "micro_steps": ["Cancel, then substitute the point."],
                })
            else:
                steps.append({
                    "instruction": "Now substitute the point into the simplified form.",
                    "expected_expression": s(expected),
                    "check_type": "limit",
                    "reason": "The simplified form gives the limit value.",
                    "micro_steps": [],
                })
        else:
            steps.append({
                "instruction": "Direct substitution fails, so simplify the expression first.",
                "expected_expression": s(expected),
                "check_type": "limit",
                "reason": "Algebra first, then the limit value follows.",
                "micro_steps": [],
            })
    else:
        steps.append({
            "instruction": "Substitution works — write the limit value.",
            "expected_expression": s(expected),
            "check_type": "limit",
            "reason": "A defined value at the point is the limit.",
            "micro_steps": [],
        })
    return steps


def build_derivative_plan(problem_text: str) -> list[dict] | None:
    """Step plan for a basic derivative problem."""
    text = (problem_text or "").strip().strip('"').strip("'")
    low = text.lower()
    if not re.search(r"d\s*/\s*d[x-z]|differentiate|derivative", low):
        return None
    m = re.search(r"(?:d\s*/\s*d[x-z]\s*(?:of)?|differentiate|derivative\s*(?:of)?)\s*(.+)$",
                  text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    try:
        from .verifier import _parse_single
        f = _parse_single(m.group(1))
        syms = sorted(f.free_symbols, key=str)
        if len(syms) != 1:
            return None
        x = syms[0]
        expected = sp.diff(f, x)
    except Exception:
        return None

    def s(e) -> str:
        try:
            return str(sp.nsimplify(sp.simplify(e), rational=True))
        except Exception:
            return str(sp.simplify(e))

    return [{
        "instruction": "Identify the rule this term needs (power, product, or chain).",
        "expected_expression": s(f),
        "check_type": "identity_step",
        "reason": "Naming the rule first keeps the algebra organized.",
        "micro_steps": [],
    }, {
        "instruction": "Apply the rule and write the derivative.",
        "expected_expression": s(expected),
        "check_type": "derivative",
        "reason": "Each rule transforms the term mechanically.",
        "micro_steps": ["Differentiate factor by factor."],
    }]


def build_system_plan(problem_text: str) -> list[dict] | None:
    """Step plan for a 2-equation linear system."""
    text = (problem_text or "").strip().strip('"').strip("'")
    if text.count("=") < 2:
        return None
    # Split on comma, semicolon, newline, or 'and'.
    parts = re.split(r"[,;\n]+|\band\b", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if "=" in p]
    if len(parts) < 2:
        return None
    try:
        from .verifier import _parse_single
        eqs = []
        syms: set = set()
        for p in parts[:2]:
            lhs = _parse_single(p.split("=", 1)[0])
            rhs = _parse_single(p.split("=", 1)[1])
            eqs.append(sp.Eq(lhs, rhs))
            syms |= set(lhs.free_symbols) | set(rhs.free_symbols)
        syms = sorted(syms, key=str)
        if len(syms) != 2:
            return None
        sols = sp.solve(eqs, syms, dict=True)
        if not sols:
            return None
        sol = sols[0]
    except Exception:
        return None

    def s(e) -> str:
        try:
            return str(sp.nsimplify(sp.simplify(e), rational=True))
        except Exception:
            return str(sp.simplify(e))

    x, y = syms
    return [{
        "instruction": f"Solve one equation for {x}.",
        "expected_expression": f"{x} = {s(sol[x])}" if x in sol else f"{x} = {s(sp.solve(eqs[0], x, dict=False)[0])}",
        "check_type": "equivalent_equation",
        "reason": "Isolating one variable lets us substitute.",
        "micro_steps": [],
    }, {
        "instruction": f"Substitute into the other equation and solve for {y}.",
        "expected_expression": f"{y} = {s(sol[y])}",
        "check_type": "equivalent_equation",
        "reason": "Substitution reduces the system to one variable.",
        "micro_steps": [],
    }, {
        "instruction": "Write the full solution pair.",
        "expected_expression": f"{x} = {s(sol[x])}, {y} = {s(sol[y])}",
        "check_type": "solution_set",
        "reason": "Both values together satisfy the system.",
        "micro_steps": [],
    }]


def check_step(expected_expression: str, student: str,
               check_type: str = "equivalent_expression") -> str:
    """CAS verdict for one guided step: correct|incorrect|unclear.

    check_equivalent compares only the RHS of an equation, so equation
    steps need a both-sides comparison here. Parses both sides with
    _parse_single directly; unparseable input is unclear, never correct.
    The check_type dispatches to the typed verifier; the legacy
    both-sides comparison is the fallback.
    """
    from .float_intent import clean_answer
    from .verifier import _parse_single, check_step_typed

    text = (student or "").strip()
    if not text:
        return "unclear"
    # Grade the math, not the prose around it: "so 3x = 9" is a real answer,
    # while "three x equals nine" is unreadable, never a wrong attempt.
    cleaned = clean_answer(text)
    if not cleaned:
        return "unclear"
    text = cleaned
    ct = (check_type or "equivalent_expression").strip().lower()
    if ct == "conceptual":
        # Two-pass LLM verdict; CAS cannot check these steps.
        try:
            from . import config as _config
            from . import nebius as _nebius
            if not _config.ai_configured():
                return "unclear"
            return _nebius.judge_conceptual(
                expected_expression, expected_expression, text)
        except Exception:
            return "unclear"
    if ct not in ("equivalent_expression", "equivalent_equation"):
        try:
            v = check_step_typed(expected_expression, text, ct)
            if v.status == "correct":
                return "correct"
            if v.status == "incomplete":
                return "partial"
            if v.status == "unverifiable":
                # One LLM normalizer retry before giving up: re-parse
                # its output with SymPy; never trust its math.
                try:
                    from . import config as _config
                    from . import nebius as _nebius
                    if _config.ai_configured():
                        normed = _nebius.normalize_via_llm(text)
                        if normed:
                            v2 = check_step_typed(
                                expected_expression, normed, ct)
                            if v2.status == "correct":
                                return "correct"
                            if v2.status == "incomplete":
                                return "partial"
                            if v2.status != "unverifiable":
                                return "incorrect"
                except Exception:
                    pass
                return "unclear"
            return "incorrect"
        except Exception:
            return "unclear"

    def sides(expr: str):
        if "=" in expr:
            lhs_raw, rhs_raw = expr.split("=", 1)
        else:
            lhs_raw, rhs_raw = expr, expr
        return _parse_single(lhs_raw), _parse_single(rhs_raw)

    try:
        s_lhs, s_rhs = sides(text)
        e_lhs, e_rhs = sides(expected_expression)
    except Exception:
        return "unclear"
    try:
        from .verifier import _items_equal

        if _items_equal(s_lhs, e_lhs) and _items_equal(s_rhs, e_rhs):
            return "correct"
        if _items_equal(s_lhs, e_rhs) and _items_equal(s_rhs, e_lhs):
            return "correct"  # swapped sides: "3x = 9" for "9 = 3x"
        # Bare value matching one side pair ("3" for "x = 3").
        if "=" not in text:
            if _items_equal(s_lhs, e_rhs) or _items_equal(s_lhs, e_lhs):
                return "correct"
        return "incorrect"
    except Exception:
        return "unclear"


def build_plan(problem_text: str, allow_llm: bool = True
               ) -> tuple[list[dict] | None, str]:
    """Dispatch across SymPy templates, then the LLM fallback.

    source is 'linear' | template name | 'llm' | 'none'. Templates run
    first; the LLM generates a plan only for uncovered problems (max 3
    tries, each validated before use). Set allow_llm=False for
    deterministic offline runs (tests). build_linear_plan stays
    byte-identical.
    """
    text = re.sub(r"(?i)^\s*(solve|find|simplify|evaluate|factor|"
                    r"differentiate|integrate|prove|work on|try)\s*[:\-]?\s*",
                    "", problem_text or "").strip()
    plan = build_linear_plan(text)
    if plan:
        return plan, "linear"
    for name, fn in (
        ("quadratic", build_quadratic_plan),
        ("trig_quad", build_trig_quad_plan),
        ("limit", build_limit_plan),
        ("derivative", build_derivative_plan),
        ("system", build_system_plan),
    ):
        try:
            plan = fn(text)
        except Exception:
            plan = None
        if plan:
            return plan, name
    if allow_llm:
        from . import config as _config
        if _config.ai_configured():
            from . import nebius as _nebius
            for _ in range(3):
                try:
                    llm_plan = _nebius.generate_plan(problem_text)
                except Exception:
                    llm_plan = None
                if not llm_plan:
                    continue
                steps = llm_plan.get("steps", []) or []
                ok, _ = validate_plan(steps, text)
                if ok:
                    return steps, "llm"
    return None, "none"


def validate_plan(plan: list[dict], problem_text: str) -> tuple[bool, str]:
    """Independent CAS validation: re-solve and compare final answers.

    Returns (ok, reason). Never trusts the plan's own math. The re-solve is
    deterministic-only: routing it back through the LLM fallback made this
    function call itself for every uncovered problem and never returned.
    """
    if not plan or not isinstance(plan, list) or len(plan) < 1:
        return False, "empty plan"
    last = plan[-1]
    check = last.get("expected_expression", "") or ""
    # Re-solve: the final answer must be re-derivable without the model.
    try:
        from .verifier import canonical_answer, check_equivalent
        redispatched, _ = build_plan(problem_text, allow_llm=False)
        if redispatched:
            re_derived = redispatched[-1].get("expected_expression", "") or ""
        else:
            answers = canonical_answer(problem_text)
            if not answers:
                return False, "could not re-solve"
            re_derived = answers[0]
        v = check_equivalent(re_derived, check)
        if v.status == "correct":
            return True, "final answer re-derived"
        return False, f"final mismatch: {v.detail}"
    except Exception as exc:
        return False, f"validator error: {exc}"


def detect_equation(text: str) -> str | None:
    """Extract a bare equation from a chat message, if one is present."""
    t = (text or "").strip()
    # Strip common lead-ins.
    t = re.sub(r"(?i)^(solve|help with|how (do|to).{0,20}?|work on|try)\s*[:\-]?\s*", "", t).strip()
    if "=" not in t:
        return None
    # Keep the equation-shaped span.
    m = re.search(r"[0-9a-zA-Z\+\-\*\/\^\(\)\.\s_x×·−]+\=[0-9a-zA-Z\+\-\*\/\^\(\)\.\s_x×·−]+", t)
    cand = (m.group(0) if m else t).strip().strip('"').strip("'")
    return cand if build_linear_plan(cand) else None
