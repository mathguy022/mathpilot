"""Practice packages: which lessons appear, and how a set of five is built.

Covers the three backend promises of the practice rebuild — every assigned
lesson is practisable (published or not), the set is topped up to five with
verified generated questions cached per lesson, and the student's place is
saved without ever shipping an answer to the browser.
"""
import os

os.environ["DEMO_MODE"] = "true"

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.db import Lesson, SessionLocal, TutorThread  # noqa: E402
from app.main import app  # noqa: E402
from tests._identity import new_teacher_client  # noqa: E402
from tests._tutor_state import wipe_tutor_state  # noqa: E402

client = TestClient(app)
# Authoring is instructor-only; student reads stay anonymous.
teacher = new_teacher_client()

TOP_UP_ANSWER = "x=7"


def _verified(prompt):
    """One question shaped like generate_questions() returns (already
    through the SymPy gate)."""
    return {
        "id": 0, "prompt_latex": prompt, "difficulty": 2,
        "steps": [{"step_id": "s1", "description": "Isolate x.",
                   "expected_input_latex": TOP_UP_ANSWER,
                   "accepted_equivalents": [], "hints": ["Undo the +3."],
                   "objective_tags": []}],
        "final_answer_latex": TOP_UP_ANSWER,
        "policy": {}, "source": "ai-generated", "is_verified": True,
    }


def _lesson_payload(title, topic, practice_math):
    sections = [
        {"kind": "explanation", "heading": "Idea",
         "body_markdown": "Solve $x+3=10$.", "math_latex": [], "examples": []},
        {"kind": "practice", "heading": "Try",
         "body_markdown": "Solve each.", "math_latex": practice_math,
         "examples": []},
    ]
    return {"title": title, "course": "Algebra", "topic": topic,
            "objectives": ["Solve linear equations"], "prerequisites": [],
            "sections": sections}


def _make_lesson(topic, practice_math, publish):
    payload = _lesson_payload(f"Practice set {topic}", topic, practice_math)
    r = teacher.post("/api/lessons/save", json={
        "course_id": 1, "topic": topic, "lesson": payload, "settings": {}})
    assert r.status_code == 200, r.text
    lid = r.json()["id"]
    a = teacher.post(f"/api/lessons/{lid}/approve", json={"lesson": payload})
    assert a.status_code == 200, a.text
    ref = None
    if publish:
        p = teacher.post("/api/publications", json={"lesson_id": lid})
        assert p.status_code == 200, p.text
        ref = p.json()["link_id"]
    else:
        ref = f"lesson-{lid}"
    return lid, ref


def _cleanup(lesson_ids, refs):
    db = SessionLocal()
    try:
        for lid in lesson_ids:
            les = db.get(Lesson, lid)
            if les:
                db.delete(les)
        db.commit()
    finally:
        db.close()
    for ref in refs:
        if not ref.startswith("lesson-"):
            teacher.delete(f"/api/publications/{ref}")


def test_approved_unpublished_lesson_is_practisable():
    lid, ref = _make_lesson("prac-unpub", ["x+3=10"], publish=False)
    try:
        page = client.get("/practice")
        assert page.status_code == 200
        assert f'value="{ref}"' in page.text
        r = client.post("/api/practice/items", json={"ref": ref})
        assert r.status_code == 200, r.text
        assert [i["item_id"] for i in r.json()["items"]][:1] == ["practice-1"]
    finally:
        _cleanup([lid], [ref])


def test_draft_lesson_never_reaches_a_student():
    payload = _lesson_payload("Draft only", "prac-draft", ["x+1=2"])
    lid = teacher.post("/api/lessons/save", json={
        "course_id": 1, "topic": "prac-draft", "lesson": payload,
        "settings": {}}).json()["id"]
    try:
        page = client.get("/practice")
        assert f'value="lesson-{lid}"' not in page.text
        assert client.post("/api/practice/items",
                           json={"ref": f"lesson-{lid}"}).status_code == 404
    finally:
        _cleanup([lid], [])


def test_items_top_up_to_five_and_cache_answers_server_side():
    lid, ref = _make_lesson("prac-topup", ["x+3=10", "x+4=11"],
                            publish=True)
    calls = {"n": 0}

    def fake_generate(**kwargs):
        calls["n"] += 1
        return {"verified": [_verified(f"Solve $x+{k}=12$.")
                             for k in range(kwargs["num_questions"])],
                "discarded": 0, "discard_reasons": [], "truncated_notes": False}

    try:
        with patch("app.generate.generate_questions", side_effect=fake_generate), \
             patch("app.config.ai_configured", return_value=True):
            first = client.post("/api/practice/items", json={"ref": ref})
            assert first.status_code == 200, first.text
            body = first.json()
            lesson_items = [i for i in body["items"] if i["source"] == "lesson"]
            generated = [i for i in body["items"] if i["source"] == "generated"]
            assert len(lesson_items) == 2
            assert len(generated) == 3, "shortfall must be filled to five"
            assert calls["n"] == 1
            # Nothing that reads like an answer key leaves the server.
            assert TOP_UP_ANSWER not in first.text
            for item in body["items"]:
                # "task" is the instruction the item was set under ("Solve for
                # x"), never its answer — the whole-body check above is what
                # keeps an answer key off the wire.
                assert set(item) <= {"item_id", "prompt_latex", "context",
                                     "hints", "hints_enabled", "source",
                                     "task"}
            # Second visit is served from the lesson cache: no new AI call.
            second = client.post("/api/practice/items", json={"ref": ref})
            assert [i["item_id"] for i in second.json()["items"]] == \
                [i["item_id"] for i in body["items"]]
            assert calls["n"] == 1
    finally:
        _cleanup([lid], [ref])


def test_a_tutor_reply_never_pays_for_a_generation_round():
    """One student message used to run the whole item generator.

    Both /api/practice/attempt and /api/tutor/message resolved the item through
    _practice_set, which tops the set up to five. A lesson of conceptual items
    fails the SymPy gate every time, so the shortfall never closed and *every*
    message paid ~15 seconds of model time. The browser gives up at 60s and the
    student sees "the tutor is still working on that — send it once more".
    """
    lid, ref = _make_lesson("prac-nogen", ["x+3=10"], publish=True)
    calls = {"n": 0}

    def fake_generate(**kwargs):
        calls["n"] += 1
        return {"verified": [], "discarded": 0, "discard_reasons": [],
                "truncated_notes": False}

    try:
        with patch("app.generate.generate_questions",
                   side_effect=fake_generate), \
             patch("app.config.ai_configured", return_value=True):
            listed = client.post("/api/practice/items", json={"ref": ref})
            assert listed.status_code == 200, listed.text
            assert calls["n"] == 1, "listing a set may top it up once"
            # The empty round is remembered, so the next visit is instant.
            client.post("/api/practice/items", json={"ref": ref})
            assert calls["n"] == 1, "a failed top-up must not repeat at once"

            item_id = listed.json()["items"][0]["item_id"]
            for url, payload in [
                ("/api/practice/attempt",
                 {"ref": ref, "item_id": item_id, "response": "x=7",
                  "course": 1}),
                ("/api/tutor/message",
                 {"message": "x=7",
                  "pageContext": {"type": "practice", "ref": ref,
                                  "item_id": item_id, "mode": "practice",
                                  "course": 1}}),
            ]:
                r = client.post(url, json=payload)
                assert r.status_code == 200, r.text
                assert calls["n"] == 1, f"{url} ran the item generator"
    finally:
        _cleanup([lid], [ref])


def test_generated_item_is_graded_from_the_cached_answer():
    lid, ref = _make_lesson("prac-grade", ["x+3=10"], publish=True)

    def fake_generate(**kwargs):
        return {"verified": [_verified(r"Solve $x+5=12$.")],
                "discarded": 0, "discard_reasons": [], "truncated_notes": False}

    try:
        with patch("app.generate.generate_questions", side_effect=fake_generate), \
             patch("app.config.ai_configured", return_value=True), \
             patch("app.nebius.float_chat", return_value=None):
            items = client.post("/api/practice/items",
                                json={"ref": ref}).json()["items"]
            gen = next(i for i in items if i["source"] == "generated")
            ok = client.post("/api/practice/attempt", json={
                "ref": ref, "item_id": gen["item_id"], "response": "x=7"})
            assert ok.status_code == 200, ok.text
            assert ok.json()["verdict"] == "correct", ok.text
            bad = client.post("/api/practice/attempt", json={
                "ref": ref, "item_id": gen["item_id"], "response": "x=9"})
            assert bad.json()["verdict"] == "incorrect", bad.text
    finally:
        _cleanup([lid], [ref])


def test_progress_autosaves_draft_and_survives_reload():
    lid, ref = _make_lesson("prac-save", ["x+3=10"], publish=True)
    try:
        r = client.post("/api/practice/progress", json={
            "ref": ref, "item_id": "practice-1", "draft": "x = 10 - 3"})
        assert r.status_code == 200, r.text
        body = client.post("/api/practice/items", json={"ref": ref}).json()
        saved = body["progress"]["practice-1"]
        assert saved["draft"] == "x = 10 - 3"
        assert saved["status"] == "started"
        skip = client.post("/api/practice/progress", json={
            "ref": ref, "item_id": "practice-1", "status": "skipped"})
        assert skip.json()["status"] == "skipped"
        assert client.post("/api/practice/progress",
                           json={"ref": ref}).status_code == 400
    finally:
        _cleanup([lid], [ref])


def test_hint_ladder_survives_a_restart():
    lid, ref = _make_lesson("prac-hints", ["x+3=10"], publish=True)
    try:
        with patch("app.nebius.float_chat", return_value=None):
            first = client.post("/api/practice/attempt", json={
                "ref": ref, "item_id": "practice-1",
                "response": "I'm stuck — give me a hint"})
            assert first.json()["hintLevel"] == 1, first.text
            # The rung the student reached is a row, not process memory: read
            # it back through a separate session, the way a new worker would.
            db = SessionLocal()
            try:
                row = db.query(TutorThread).filter(
                    TutorThread.key.startswith(
                        f"practice:{ref}:practice-1:")).first()
                assert row is not None and row.hints == 1, row
            finally:
                db.close()
            second = client.post("/api/practice/attempt", json={
                "ref": ref, "item_id": "practice-1",
                "response": "another hint please"})
            assert second.json()["hintLevel"] == 2, second.text
    finally:
        wipe_tutor_state(f"practice:{ref}:")
        _cleanup([lid], [ref])
