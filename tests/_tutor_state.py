"""Wipe the tutor state the app now stores in SQLite.

These were process-memory dicts the tests cleared with `.clear()`/`.pop()`.
The threads, the pending "start a new problem?" offers and the rate-limit
window live in tables now, so measuring a hint ladder from zero means
deleting rows.
"""
from app.db import SessionLocal, TutorPending, TutorRateHit, TutorThread


def wipe_tutor_state(key_prefix: str = "") -> None:
    """Drop stored threads — all of them, or only those under `key_prefix`."""
    db = SessionLocal()
    try:
        threads = db.query(TutorThread)
        if key_prefix:
            threads = threads.filter(TutorThread.key.startswith(key_prefix))
        threads.delete(synchronize_session=False)
        db.query(TutorPending).delete(synchronize_session=False)
        db.query(TutorRateHit).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()
