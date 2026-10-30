"""Regression tests for the practice panel's AI tutor chat.

The panel used to grade every message as a step answer, so greetings and
questions collapsed into one repeated "I couldn't read that" line. Same
contract as the floating widget: the router decides whether a message is an
attempt, SymPy decides correctness, the LLM only words the reply, and hints
walk a server-side ladder.

Run: py -m pytest tests/test_practice_tutor.py -q
"""
import os

os.environ["DEMO_MODE"] = "true"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app.db import SessionLocal, TutorEvent, TutorSession  # noqa: E402
from app.main import app  # noqa: E402
from tests._identity import new_teacher_client  # noqa: E402
from tests._tutor_state import wipe_tutor_state  # noqa: E402

client = TestClient(app)
# Authoring is instructor-only; student reads stay anonymous.
teacher = new_teacher_client()

# The practice section's math_latex becomes both the student prompt and the
# CAS target, so "3(x-3)" verifies correct against "3x-9" and "3x+9" does not.
LESSON = {
    "title": "Practice tutor lesson", "course": "Algebra 1",
    "topic": "Factoring", "objectives": ["Factor a linear expression"],
    "prerequisites": ["Algebra"],
    "sections": [
        {"kind": "explanation", "heading": "Idea",
         "body_markdown": "Factor out the common term.",
         "math_latex": [], "examples": []},
        {"kind": "practice", "heading": "Try",
         "body_markdown": "Factor.",
         "math_latex": ["3x-9", "3x-9=0", "\\lim_{x \\to 1} x"],
         "examples": []},
        {"kind": "solutions", "heading": "Keys",
         "body_markdown": "Answer: $3(x-3)$.",
         "math_latex": [], "examples": []},
    ],
}
ITEM = "practice-1"
_course = "Algebra 1"


def _attempt(text, item=ITEM):
    return client.post("/api/practice/attempt", json={
        "ref": _pub[1], "item_id": item, "response": text,
        "course": _course}).json()


_pub = None


def _publish():
    r = teacher.post("/api/lessons/save", json={
        "course_id": 1, "topic": LESSON["topic"], "lesson": LESSON,
        "settings": {}})
    lid = r.json()["id"]
    teacher.post(f"/api/lessons/{lid}/approve", json={"lesson": LESSON})
    p = teacher.post("/api/publications", json={
        "lesson_id": lid, "title": "Practice tutor pkg"})
    assert p.status_code == 200, p.text
    return lid, p.json()["link_id"]


def _clear_saved_progress(link_id):
    """The ladder and the transcript are stored rows now, so wipe those too."""
    from app.db import PracticeProgress

    wipe_tutor_state(f"practice:{link_id}:")
    db = SessionLocal()
    try:
        db.query(PracticeProgress).filter(
            PracticeProgress.package_ref == link_id).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture(scope="module")
def _deck():
    global _pub
    lid, link_id = _publish()
    _pub = (lid, link_id)
    yield _pub
    db = SessionLocal()
    try:
        db.query(TutorEvent).delete()
        db.query(TutorSession).delete()
        db.commit()
    finally:
        db.close()
    teacher.delete(f"/api/publications/{link_id}")
    teacher.delete(f"/api/lessons/{lid}")


@pytest.fixture(autouse=True)
def _clean(_deck):
    """No network, and every test starts with an untouched hint ladder."""
    link_id = _deck[1]
    _clear_saved_progress(link_id)
    with patch("app.config.ai_configured", return_value=True), \
         patch("app.nebius.float_chat", return_value=None), \
         patch("app.nebius.generate_plan", return_value=None), \
         patch("app.nebius.normalize_via_llm", return_value=None), \
         patch("app.nebius.judge_conceptual", return_value="unclear"):
        yield
    _clear_saved_progress(link_id)


def test_prose_is_not_graded_and_burns_no_hint():
    body = _attempt("hi")
    assert body["verdict"] == "not_an_attempt", body
    assert body["hintLevel"] == 0, body
    assert body["misses"] == 0, body
    assert body["reply"].strip()


def test_a_real_question_is_not_graded_as_an_attempt():
    body = _attempt("what does factoring even mean?")
    assert body["verdict"] == "not_an_attempt", body
    assert body["hintLevel"] == 0, body


def test_wrong_answers_walk_the_ladder_instead_of_repeating():
    first = _attempt("3x+9")
    second = _attempt("3x-8")
    assert first["verdict"] == second["verdict"] == "incorrect"
    assert (first["hintLevel"], second["hintLevel"]) == (1, 2)
    assert first["reply"].startswith("Hint 1 —"), first
    assert second["reply"].startswith("Hint 2 —"), second
    assert first["replyType"] == "hint"
    assert second["misses"] == 2


def test_cas_decides_correctness_even_when_the_model_praises():
    with patch("app.nebius.float_chat",
               return_value="Amazing, you nailed it!"):
        body = _attempt("3x+9")
    assert body["verdict"] == "incorrect", body
    assert body["misses"] == 1, body


def test_correct_attempt_verifies_and_resets_the_ladder():
    _attempt("3x+9")
    body = _attempt("3(x-3)")
    assert body["verdict"] == "correct", body
    assert body["hintLevel"] == 0, body
    assert body["replyType"] == "success"


def test_asked_for_the_answer_it_refuses_without_revealing():
    body = _attempt("just tell me the answer")
    assert body["verdict"] == "not_an_attempt", body
    assert body["revealed"] is False, body
    assert body["reply"].startswith("I can't give the final answer"), body
    assert "3(x-3)" not in body["reply"]


def test_solved_equation_item_grades_the_solution_not_the_problem_text():
    # The item's stored math is "3x-9=0"; the answer is x=3. Grading an
    # attempt against the problem text made every correct response wrong.
    assert _attempt("x=3", "practice-2")["verdict"] == "correct"
    wrong = _attempt("x=5", "practice-2")
    assert wrong["verdict"] == "incorrect", wrong
    assert wrong["hintLevel"] == 1, wrong


def test_limit_item_grades_the_limit_value():
    assert _attempt("1", "practice-3")["verdict"] == "correct"
    assert _attempt("0", "practice-3")["verdict"] == "incorrect"


def test_a_reply_that_states_the_answer_is_replaced_by_the_safe_line():
    # Live failure: the hint read "For just x, the answer is 1".
    with patch("app.nebius.float_chat",
               return_value="For just x, the answer is 1."):
        body = _attempt("I'm stuck — give me a hint", "practice-3")
    assert body["verdict"] == "hint", body
    assert "answer is 1" not in body["reply"], body
    assert body["reply"].startswith("Hint 1 —"), body


def test_quoting_the_problem_is_not_censored_as_a_reveal():
    with patch("app.nebius.float_chat",
               return_value="As x approaches 1, what does x approach?"):
        body = _attempt("I'm stuck — give me a hint", "practice-3")
    assert body["reply"].endswith("what does x approach?"), body


def test_hint_that_solves_the_equation_is_replaced_by_the_safe_line():
    # Live failure: the hint did the algebra and ended on "x = 3".
    with patch("app.nebius.float_chat",
               return_value="Substitute to check: 3(5)-9 = 6, not 0. "
                            "Isolate x: 3x = 9, so x = 3."):
        body = _attempt("I'm stuck — give me a hint", "practice-2")
    assert body["reply"].startswith("Hint 1 —"), body
    assert "x = 3" not in body["reply"], body
    assert "so x" not in body["reply"], body


def test_canonical_answer_only_covers_shapes_sym_py_solves():
    from app.verifier import canonical_answer

    assert canonical_answer(r"\lim_{x \to 1} x") == ["1"]
    assert canonical_answer("3x-9=0") == ["3"]
    assert canonical_answer("x^2-5x+6=0") == ["2", "3"]
    # A bare expression has no single answer: guessing one would censor
    # correct guidance on factoring items.
    assert canonical_answer("3x-9") == []
    assert canonical_answer("explain the chain rule") == []


def test_the_model_cannot_print_its_own_hint_label():
    """Live failure: the reply read "Hint 1 — Hint 2 (of 3): …" — two labels,
    and a rung number that disagreed with the ladder the student was on."""
    with patch("app.nebius.float_chat",
               return_value="Hint 2 (of 3): Look at the common factor."):
        body = _attempt("I'm stuck — give me a hint")
    assert body["verdict"] == "hint", body
    assert body["reply"] == "Hint 1 — Look at the common factor.", body


def test_help_request_gives_a_hint_and_remembers_the_level():
    first = _attempt("I'm stuck, give me a hint")
    second = _attempt("another hint please")
    assert first["verdict"] == second["verdict"] == "hint"
    assert first["hintLevel"] == 1 and second["hintLevel"] == 2


def test_empty_message_prompts_for_input_without_a_verdict():
    body = client.post("/api/practice/attempt", json={
        "ref": _pub[1], "item_id": ITEM, "response": "   "}).json()
    assert body["verdict"] == "incomplete", body
    assert "step by step" in body["reply"]


def test_unparseable_attempt_asks_for_restatement_not_a_grade():
    body = _attempt("um idk it's a thing")
    assert body["verdict"] == "not_an_attempt", body
    assert body["misses"] == 0, body


def test_step_by_step_handoff_opens_a_session_on_the_item():
    # Mirrors what the "Work it step by step" button posts: the item's own
    # prompt, with no prose around it — a prefix breaks plan building.
    db = SessionLocal()
    try:
        db.query(TutorSession).delete()
        db.commit()
    finally:
        db.close()
    r = client.post("/api/tutor/message", json={
        "message": "$$3x-9=0$$",
        "pageContext": {"type": "general", "courseId": "practice-handoff",
                        "course": _course, "mode": "practice"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["replyType"] == "step", body
    assert body["totalSteps"] >= 1, body
    db = SessionLocal()
    try:
        assert db.query(TutorSession).count() == 1
    finally:
        db.close()


def test_practice_content_never_ships_the_solution_section():
    page = client.get(f"/learn/{_pub[1]}")
    assert page.status_code == 200
    assert "Answer: $3(x-3)$" not in page.text


def test_retyping_the_item_is_not_marked_correct():
    """The item text came back as "verified" for expression items.

    "3x-9" is graded by equivalence, so the question and a copy of it look
    the same to the checker; only an exact restatement is refused, so a real
    rewrite still verifies.
    """
    body = _attempt("3x-9")
    assert body["verdict"] == "not_an_attempt", body
    assert (body["misses"], body["hintLevel"]) == (0, 0), body
    assert "first move" in body["reply"], body
    assert _attempt("3(x-3)")["verdict"] == "correct"


QUAD_LESSON = {
    "title": "Quadratic practice tutor lesson", "course": "Algebra 2",
    "topic": "Quadratics", "objectives": ["Solve by factoring"],
    "prerequisites": ["Factoring"],
    "sections": [
        {"kind": "explanation", "heading": "Idea",
         "body_markdown": "Factor, then set each factor to zero.",
         "math_latex": [], "examples": []},
        {"kind": "practice", "heading": "Try",
         "body_markdown": "Solve.",
         "math_latex": ["x^2-7x+10=0"], "examples": []},
        {"kind": "solutions", "heading": "Keys",
         "body_markdown": "Answer: $x=2, x=5$.",
         "math_latex": [], "examples": []},
    ],
}


@pytest.fixture(scope="module")
def _quad():
    r = teacher.post("/api/lessons/save", json={
        "course_id": 1, "topic": QUAD_LESSON["topic"], "lesson": QUAD_LESSON,
        "settings": {}})
    lid = r.json()["id"]
    teacher.post(f"/api/lessons/{lid}/approve", json={"lesson": QUAD_LESSON})
    p = teacher.post("/api/publications", json={
        "lesson_id": lid, "title": "Quadratic practice pkg"})
    assert p.status_code == 200, p.text
    link = p.json()["link_id"]
    yield lid, link
    _clear_saved_progress(link)
    teacher.delete(f"/api/publications/{link}")
    teacher.delete(f"/api/lessons/{lid}")


def test_stuck_questions_walk_the_verified_plan_instead_of_repeating(_quad):
    """Live failure: "what should I do?" on a quadratic got the linear
    playbook ("move the number to the right"), then the same sentence again,
    then Hint 4 repeating itself. Guidance must follow the item's own verified
    solution path, one distinct move per ask, and never state a root."""
    link = _quad[1]
    _clear_saved_progress(link)

    def ask():
        return client.post("/api/practice/attempt", json={
            "ref": link, "item_id": "practice-1",
            "response": "What should I do ?", "course": _course}).json()

    first, second, third = ask(), ask(), ask()
    assert first["replyType"] == second["replyType"] == "step"
    assert "factored form" in first["reply"], first
    assert "Set each factor equal" in second["reply"], second
    assert "Check one solution" in third["reply"], third
    assert len({first["reply"], second["reply"], third["reply"]}) == 3
    # Asking where to start is not a hint request: the ladder stays put.
    assert first["hintLevel"] == second["hintLevel"] == third["hintLevel"] == 0
    for body in (first, second, third):
        flat = body["reply"].replace(" ", "")
        assert "x=2" not in flat and "x=5" not in flat, body
    # Once the verified path is used up, another "what should I do?" must not
    # replay its last move: the hint ladder takes over instead.
    fourth = ask()
    assert fourth["replyType"] == "hint", fourth
    assert fourth["hintLevel"] == 1, fourth
    assert fourth["reply"].startswith("Hint 1 —"), fourth
    _clear_saved_progress(link)


def test_thread_state_persists_the_move_counter():
    """The counter of moves already handed out has to survive a reload.

    It lived only in the in-memory state dict, so every request read it back
    as zero and the plan walk restarted at move 1 however far it had got.
    """
    from app.db import SessionLocal, TutorThread
    from app.main import _thread

    key = "practice:probe:moves:u0"
    db = SessionLocal()
    try:
        thread = _thread(db, key)
        thread.state["moves"] = 2
        thread.save()
        assert _thread(db, key).state["moves"] == 2
    finally:
        db.query(TutorThread).filter(TutorThread.key == key).delete(
            synchronize_session=False)
        db.commit()
        db.close()


def test_a_model_that_repeats_itself_still_walks_the_plan(_quad):
    """Live failure: the wording model sent its first move three times in a
    row, so the walk looked stuck on step 1 even though the move counter had
    advanced. A reply that repeats, or that is not worded in the plan's own
    method, is replaced by the move this turn is actually on."""
    link = _quad[1]
    _clear_saved_progress(link)

    def ask():
        return client.post("/api/practice/attempt", json={
            "ref": link, "item_id": "practice-1",
            "response": "What should I do ?", "course": _course}).json()

    with patch("app.nebius.float_chat", return_value="Try factoring it."):
        first, second, third = ask(), ask(), ask()
    replies = [b["reply"] for b in (first, second, third)]
    assert len(set(replies)) == 3, replies
    # "Try factoring it." names no move, so all three come from the plan.
    assert "factored form" in first["reply"], first
    assert "Set each factor equal" in second["reply"], second
    assert "Check one solution" in third["reply"], third
    _clear_saved_progress(link)


def test_a_hint_worded_for_another_method_is_replaced_by_the_plan_move(_quad):
    """Live failure: rung 1 on x^2-7x+10=0 came back as "whatever you do to
    one side, do to the other" — a linear move that cannot finish a factoring
    item, so the student is sent down a dead end."""
    link = _quad[1]
    _clear_saved_progress(link)
    with patch("app.nebius.float_chat",
               return_value="Whatever you do to one side, do to the other."):
        body = client.post("/api/practice/attempt", json={
            "ref": link, "item_id": "practice-1",
            "response": "give me a hint", "course": _course}).json()
    assert body["verdict"] == "hint", body
    assert body["reply"].startswith("Hint 1 —"), body
    assert "factored form" in body["reply"], body
    assert "one side" not in body["reply"], body
    _clear_saved_progress(link)


def test_a_hint_that_states_the_factored_form_is_replaced(_quad):
    """Live failure: "(x - 2)(x - 5) = 0" hands over both roots without ever
    using the word answer, so the reveal gate waved it through."""
    link = _quad[1]
    _clear_saved_progress(link)
    with patch("app.nebius.float_chat",
               return_value="The factoring step is: (x - 2)(x - 5) = 0."):
        body = client.post("/api/practice/attempt", json={
            "ref": link, "item_id": "practice-1",
            "response": "give me a hint", "course": _course}).json()
    assert body["verdict"] == "hint", body
    assert body["reply"].startswith("Hint 1 —"), body
    flat = body["reply"].replace(" ", "")
    assert "(x-2)" not in flat and "(x-5)" not in flat, body
    assert "factored form" in body["reply"], body
    _clear_saved_progress(link)


def test_a_label_the_model_copies_mid_reply_is_removed():
    """Live failure: "Hint 2 — That's not the factored form yet. Hint 2 — …"
    showed the student two labels, one of them a rung they are not on."""
    with patch("app.nebius.float_chat",
               return_value="That is not it yet. Hint 2 — Look at the 9."):
        body = _attempt("3x+9")
    assert body["verdict"] == "incorrect", body
    assert body["reply"].count("Hint") == 1, body
    assert body["reply"].startswith("Hint 1 — That is not it yet."), body


def test_hint_on_a_quadratic_names_a_move_from_the_verified_plan(_quad):
    link = _quad[1]
    _clear_saved_progress(link)
    body = client.post("/api/practice/attempt", json={
        "ref": link, "item_id": "practice-1",
        "response": "I'm stuck — give me a hint", "course": _course}).json()
    assert body["verdict"] == "hint", body
    assert body["reply"].startswith("Hint 1 —"), body
    assert "factored form" in body["reply"], body
    _clear_saved_progress(link)


LIN_LESSON = {
    "title": "Decimal linear practice tutor lesson", "course": "Algebra 2",
    "topic": "Linear Equations", "objectives": ["Solve a decimal linear equation"],
    "prerequisites": ["Like terms"],
    "sections": [
        {"kind": "explanation", "heading": "Idea",
         "body_markdown": "Open the bracket, collect, isolate.",
         "math_latex": [], "examples": []},
        {"kind": "practice", "heading": "Try",
         "body_markdown": "Solve.",
         "math_latex": ["0.4x - 0.2(3 - x) = 1.8"], "examples": []},
        {"kind": "solutions", "heading": "Keys",
         "body_markdown": "Answer: $x=4$.",
         "math_latex": [], "examples": []},
    ],
}


@pytest.fixture(scope="module")
def _lin():
    r = teacher.post("/api/lessons/save", json={
        "course_id": 1, "topic": LIN_LESSON["topic"], "lesson": LIN_LESSON,
        "settings": {}})
    lid = r.json()["id"]
    teacher.post(f"/api/lessons/{lid}/approve", json={"lesson": LIN_LESSON})
    p = teacher.post("/api/publications", json={
        "lesson_id": lid, "title": "Decimal linear practice pkg"})
    assert p.status_code == 200, p.text
    link = p.json()["link_id"]
    yield lid, link
    _clear_saved_progress(link)
    teacher.delete(f"/api/publications/{link}")
    teacher.delete(f"/api/lessons/{lid}")


def _lin_ask(link, text):
    return client.post("/api/practice/attempt", json={
        "ref": link, "item_id": "practice-1", "response": text,
        "course": _course}).json()


def test_first_move_names_a_term_the_student_can_see(_lin):
    """Live failure: the item 0.4x - 0.2(3 - x) = 1.8 was planned as
    "Add 3/5 to both sides", but the -3/5 only exists once the bracket is
    multiplied out, so the first move asked about a term that is not in the
    problem — and the wording model made it worse ("add it to the left side
    only", which is not a legal move)."""
    link = _lin[1]
    _clear_saved_progress(link)
    body = _lin_ask(link, "What should I do first?")
    assert body["replyType"] == "step", body
    assert "bracket" in body["reply"].lower(), body
    assert "3/5" not in body["reply"], body
    assert "x=4" not in body["reply"].replace(" ", ""), body
    _clear_saved_progress(link)


def test_a_correct_intermediate_line_is_not_a_miss(_lin):
    """The item is graded against its own solution, so 0.6x = 2.4 — exactly
    what move 2 of the verified path asked for — came back "incorrect" and
    cost the student a miss for doing what the tutor just requested."""
    link = _lin[1]
    _clear_saved_progress(link)
    _lin_ask(link, "What should I do first?")
    body = _lin_ask(link, "0.6x = 2.4")
    assert body["verdict"] == "partial", body
    assert body["misses"] == 0, body
    assert body["replyType"] == "step", body
    assert body["hintLevel"] == 0, body
    flat = body["reply"].replace(" ", "")
    assert "x=4" not in flat, body
    _clear_saved_progress(link)
