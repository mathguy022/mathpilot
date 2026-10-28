"""Instructor-facing insights, aggregated from the tutor's own event log.

Every rate here has a denominator the instructor can see, and only covers
steps the CAS actually decided. Anything the checker could not decide is
counted separately instead of being folded into a percentage, so an insight
never looks stronger than the evidence behind it. Rows are aggregated: no
student is identified to the instructor.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .db import Attempt, HintRequest, TutorEvent, TutorSession

# The checker reached a verdict on the student's own work.
DECIDED = ("correct", "incorrect", "partial")
# The student wrote something the checker could not grade.
UNDECIDED = ("unclear", "unverifiable", "not_an_attempt", "incomplete")
# Asking for the answer, and being turned down.
SEEKING = ("refused",)
# The student asked for help; the reply was a rung of the hint ladder.
ASKING = ("hint",)
# Ran out of the hint ladder before finishing.
CAPPED = ("hint-capped",)
MAX_TEXT = 60


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def _pl(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _trim(text: str) -> str:
    t = " ".join((text or "").split())
    return t[:MAX_TEXT] + ("…" if len(t) > MAX_TEXT else "")


def build(db, course: str | None = None) -> dict:
    """Return the insights payload for one course, or all courses together."""
    sess_q = db.query(TutorSession)
    if course:
        sess_q = sess_q.filter(TutorSession.course_id == course)
    sessions = sess_q.all()
    by_id = {s.id: s for s in sessions}

    events = []
    if by_id:
        events = (db.query(TutorEvent)
                  .filter(TutorEvent.session_id.in_(list(by_id))).all())

    decided = [e for e in events if e.verdict in DECIDED]
    undecided = [e for e in events if e.verdict in UNDECIDED]
    incorrect = [e for e in decided if e.verdict == "incorrect"]
    seeking = [e for e in events if e.verdict in SEEKING]
    capped = [e for e in events if e.verdict in CAPPED]
    # A wrong step also comes back as a hint, so event_type alone would
    # count every miss as a request for help. Only the student's own asks
    # belong in the ladder.
    hints = [e for e in events if e.verdict in ASKING]
    nudges = [e for e in incorrect if e.event_type == "hint"]

    return {
        "scope": course or "all courses",
        "window": _window(sessions, events),
        "coverage": _coverage(sessions, events, decided, undecided, db),
        "courses": _courses(sessions, decided, incorrect, undecided, hints),
        "insights": _stall_points(decided, incorrect),
        "assistance": _assistance(hints, nudges, seeking, capped, sessions),
        "recurring_candidates": _candidates(incorrect, by_id),
        "limitations": _limitations(decided, undecided, seeking),
    }


def _window(sessions, events) -> dict:
    stamps = [e.created_at for e in events if e.created_at]
    stamps += [s.created_at for s in sessions if s.created_at]
    if not stamps:
        return {"since": None, "through": None}
    lo, hi = min(stamps), max(stamps)
    return {"since": str(lo)[:16], "through": str(hi)[:16]}


def _coverage(sessions, events, decided, undecided, db) -> dict:
    graded_sessions = {e.session_id for e in decided}
    return {
        "sessions": len(sessions),
        "students": len({s.student_id for s in sessions}),
        "events": len(events),
        "steps_decided": len(decided),
        "steps_undecided": len(undecided),
        "sessions_with_graded_work": len(graded_sessions),
        "legacy_activity_attempts": db.query(Attempt).count(),
        "legacy_hint_requests": db.query(HintRequest).count(),
    }


def _courses(sessions, decided, incorrect, undecided, hints) -> list[dict]:
    grouped: dict[str, dict] = defaultdict(
        lambda: {"sessions": set(), "students": set(), "decided": 0,
                 "incorrect": 0, "undecided": 0, "hints": 0, "abandoned": 0})
    for s in sessions:
        key = s.course_id or "unassigned"
        grouped[key]["sessions"].add(s.id)
        grouped[key]["students"].add(s.student_id)
        if s.status == "abandoned":
            grouped[key]["abandoned"] += 1
    course_of = {s.id: (s.course_id or "unassigned") for s in sessions}
    for bucket, rows in (("decided", decided), ("incorrect", incorrect),
                         ("undecided", undecided)):
        for e in rows:
            grouped[course_of.get(e.session_id, "unassigned")][bucket] += 1
    for e in hints:
        grouped[course_of.get(e.session_id, "unassigned")]["hints"] += 1

    out = []
    for name, g in grouped.items():
        out.append({
            "course": name,
            "sessions": len(g["sessions"]),
            "students": len(g["students"]),
            "steps_decided": g["decided"],
            "incorrect": g["incorrect"],
            "error_rate": _rate(g["incorrect"], g["decided"]),
            "undecided": g["undecided"],
            "hint_requests": g["hints"],
            "abandoned_sessions": g["abandoned"],
        })
    return sorted(out, key=lambda r: (-r["steps_decided"], r["course"]))


def _stall_points(decided, incorrect) -> list[dict]:
    """Per-step struggle, numbered from 1 as the student sees it.

    Kept under the historical 'insights' key: it is the one view the
    instructor page has always had.
    """
    seen, wrong = Counter(), Counter()
    for e in decided:
        seen[e.step_index] += 1
    for e in incorrect:
        wrong[e.step_index] += 1
    rows = [{"step": idx + 1, "seen": seen[idx], "incorrect": wrong[idx],
             "rate": _rate(wrong[idx], seen[idx]),
             "label": f"Step {idx + 1}"} for idx in seen]
    return sorted(rows, key=lambda r: (-r["rate"], -r["seen"], r["step"]))


def _assistance(hints, nudges, seeking, capped, sessions) -> dict:
    ladder = Counter(min(e.hint_level or 1, 5) for e in hints)
    return {
        "hint_requests": len(hints),
        "hint_ladder": [{"level": lv, "seen": ladder[lv]}
                        for lv in sorted(ladder)],
        "corrective_hints": len(nudges),
        "hint_capped": len(capped),
        "answer_requests": len(seeking),
        "sessions_needing_hints": len({e.session_id for e in hints}),
        "abandoned_sessions": sum(1 for s in sessions
                                  if s.status == "abandoned"),
        "completed_sessions": sum(1 for s in sessions
                                  if s.status == "completed"),
        "mean_misses_per_session": _rate(
            sum(s.total_misses or 0 for s in sessions), len(sessions)),
    }


def _candidates(incorrect, sessions_by_id) -> list[dict]:
    """Repeated wrong answers at the same step — candidates, not conclusions.

    The count is of responses, not of students: one student repeating an
    error is weaker evidence than a class sharing it, so both are reported.
    """
    grouped: dict[tuple, dict] = defaultdict(
        lambda: {"count": 0, "students": set(), "course": set()})
    for e in incorrect:
        sess = sessions_by_id.get(e.session_id)
        key = ((sess.course_id if sess else "") or "unassigned",
               e.step_index, " ".join((e.student_input or "").split()).lower())
        grouped[key]["count"] += 1
        if sess:
            grouped[key]["students"].add(sess.student_id)
            grouped[key]["course"].add(sess.course_id or "unassigned")
    out = []
    for (course, step, text), g in grouped.items():
        if not text or g["count"] < 2:
            continue
        out.append({
            "course": course,
            "step": step + 1,
            "response": _trim(text),
            "occurrences": g["count"],
            "students": len(g["students"]),
            "basis": "inferred from repeated responses",
        })
    out.sort(key=lambda r: (-r["occurrences"], -r["students"], r["step"]))
    return out[:8]


def _limitations(decided, undecided, seeking) -> list[str]:
    notes = [
        "Error rates cover only the steps SymPy could decide "
        f"({len(decided)} of them); they are not class-wide.",
    ]
    if undecided:
        n = _pl(len(undecided), "response", "responses")
        verb = "is" if len(undecided) == 1 else "are"
        notes.append(
            f"{n} could not be read as math and {verb} counted separately, "
            "never as wrong answers.")
    else:
        notes.append("No unread responses were recorded in this window.")
    notes.append(
        "Recurring candidates are repeated responses at the same step. "
        "They are leads to check against the student's full work, not "
        "evidence of a misconception.")
    if seeking:
        n = _pl(len(seeking), "request", "requests")
        verb = "was" if len(seeking) == 1 else "were"
        notes.append(
            f"{n} for the final answer {verb} declined; practice keeps the "
            "answer behind the checker.")
    notes.append("Figures are aggregated; no student is named or linked.")
    return notes
