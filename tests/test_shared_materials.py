"""Shared Materials: instructor upload, share gating, and student access."""
import os

os.environ["DEMO_MODE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, UPLOAD_DIR  # noqa: E402
from app import auth  # noqa: E402
from app.db import SessionLocal, User, SharedFile  # noqa: E402

client = TestClient(app)

# Minimal valid-enough PDF/DOCX payloads: content type is decided by the
# extension, and the server never parses the bytes, so a short body is fine.
PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


def _uid(role):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.role == role).first().id
    finally:
        db.close()


def _as(role):
    client.cookies.clear()
    client.cookies.set(auth.COOKIE, auth.issue(_uid(role)))


def _rows():
    db = SessionLocal()
    try:
        return db.query(SharedFile).all()
    finally:
        db.close()


def _cleanup(ids):
    db = SessionLocal()
    try:
        for r in db.query(SharedFile).filter(SharedFile.id.in_(ids)).all():
            p = UPLOAD_DIR / r.stored_name
            db.delete(r)
            db.commit()
            if p.is_file():
                p.unlink()
    finally:
        db.close()


def test_upload_starts_private_and_rejects_bad_type():
    _as("instructor")
    before = {r.id for r in _rows()}

    r = client.post("/instructor/materials/upload",
                    files={"file": ("notes.pdf", PDF, "application/pdf")},
                    data={"title": "Lecture 1 notes", "course": "Calculus"})
    # TestClient follows the 303 redirect to the panel, so the final status
    # is the panel's 200; the row-existence assertions below prove the upload.
    assert r.status_code in (200, 302, 303)
    new = [x for x in _rows() if x.id not in before]
    assert len(new) == 1
    row = new[0]
    assert row.shared is False
    assert row.title == "Lecture 1 notes"

    # Disallowed extension is rejected and stores nothing.
    bad = client.post("/instructor/materials/upload",
                      files={"file": ("evil.exe", b"MZ", "application/x-msdownload")})
    assert bad.status_code == 400
    assert (UPLOAD_DIR / "evil.exe").is_file() is False

    _cleanup([row.id])


def test_student_cannot_see_or_fetch_until_shared():
    _as("instructor")
    r = client.post("/instructor/materials/upload",
                    files={"file": ("notes.pdf", PDF, "application/pdf")},
                    data={"title": "Private", "course": ""})
    row = [x for x in _rows() if x.title == "Private"][0]

    # Student: not listed, and direct fetch is forbidden while unshared.
    _as("student")
    assert "Private" not in client.get("/shared").text
    assert client.get(f"/shared/file/{row.id}").status_code == 403

    # Instructor shares it.
    _as("instructor")
    client.post(f"/instructor/materials/{row.id}/share")

    # Student now sees it and can fetch it inline.
    _as("student")
    page = client.get("/shared")
    assert page.status_code == 200 and "Private" in page.text
    fetched = client.get(f"/shared/file/{row.id}")
    assert fetched.status_code == 200
    assert fetched.headers["content-type"] == "application/pdf"
    assert "inline" in fetched.headers.get("content-disposition", "")

    _cleanup([row.id])


def test_word_document_downloads_and_unshare_hides():
    _as("instructor")
    client.post("/instructor/materials/upload",
                files={"file": ("handout.docx", b"PK\x03\x04fake", "application/vnd"
                                ".openxmlformats-officedocument.wordprocessingml.document")},
                data={"title": "Handout", "course": ""})
    row = [x for x in _rows() if x.title == "Handout"][0]
    client.post(f"/instructor/materials/{row.id}/share")

    _as("student")
    fetched = client.get(f"/shared/file/{row.id}")
    assert fetched.status_code == 200
    assert "attachment" in fetched.headers.get("content-disposition", "")

    # Unshare removes it from the student view again.
    _as("instructor")
    client.post(f"/instructor/materials/{row.id}/share")
    _as("student")
    assert "Handout" not in client.get("/shared").text

    _cleanup([row.id])


def test_instructor_only_endpoints_reject_students():
    _as("student")
    assert client.get("/instructor/materials").status_code == 403
    r = client.post("/instructor/materials/upload",
                    files={"file": ("x.pdf", PDF, "application/pdf")})
    assert r.status_code == 403
