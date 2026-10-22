"""Seed the course catalog and demo users. All synthetic, no PII."""
from .courses import SUPPORTED_COURSES
from .db import SessionLocal, User, Course


def seed():
    db = SessionLocal()
    try:
        # Curriculum catalog is seeded independently so existing DBs pick
        # up new courses on restart (idempotent by title).
        existing = {c.title for c in db.query(Course).all()}
        for meta in SUPPORTED_COURSES.values():
            if meta["label"] not in existing:
                db.add(Course(title=meta["label"], topic=meta["label"]))
        # Legacy label: keep "Calculus I" working (catalog uses "Calculus 1").
        if "Calculus I" not in existing and "Calculus 1" not in existing:
            db.add(Course(title="Calculus I", topic="Calculus"))
        db.commit()

        # NOTE: practice activities, materials, and mastery rows are no
        # longer seeded. The student Assignments tab lists only
        # instructor-assigned packages; practice content previously
        # referenced by tests is created by the tests themselves.
        if db.query(User).first():
            return
        db.add_all([
            User(name="Alex Johnson", role="student"),
            User(name="Dr. Sarah Wilson", role="instructor"),
            User(name="Maria Garcia", role="student"),
            User(name="James Lee", role="student"),
        ])
        db.commit()
    finally:
        db.close()
