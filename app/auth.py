"""Demo identity: who the browser claims to be, and what that role may read.

There is no password layer yet. A request carries an HMAC-signed cookie
naming a roster user, and the role is read from that user's database row —
never from the cookie — so switching roles means switching accounts, not
editing a token. This makes the boundary real in application code instead of
in a template, but it is still demo identity: anyone who can reach the
"View as" picker can claim any account. Deployed authentication must replace
it with a verified login and a secure cookie.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import User, get_db

COOKIE = "mp_viewer"
SECRET_ENV = "MATHPILOT_SECRET"
MAX_AGE = 30 * 24 * 3600

# Without a configured secret, sessions stay valid across serverless instances
# by deriving a stable seed from DATABASE_URL or falling back to a demo secret.
def _get_secret() -> bytes:
    env_sec = os.getenv(SECRET_ENV)
    if env_sec:
        return env_sec.encode()
    from .config import DATABASE_URL
    if DATABASE_URL and "sqlite" not in DATABASE_URL:
        return hashlib.sha256(f"mathpilot-stable-secret-{DATABASE_URL}".encode()).digest()
    return b"mathpilot-demo-static-cookie-secret-key-32b"


_SECRET = _get_secret()


def _sign(body: str) -> str:
    return hmac.new(_SECRET, body.encode(), hashlib.sha256).hexdigest()


def issue(user_id: int) -> str:
    body = base64.urlsafe_b64encode(str(int(user_id)).encode()).decode()
    return f"{body}.{_sign(body)}"


def resolve(token: str | None) -> int | None:
    """Return the user id in a valid token, or None. Never raises."""
    if not token or "." not in token:
        return None
    body, _, sig = token.rpartition(".")
    if not hmac.compare_digest(_sign(body), sig):
        return None
    try:
        return int(base64.urlsafe_b64decode(body + "==").decode())
    except Exception:
        return None


def is_public_demo() -> bool:
    """True when public demo access is active (default for web/deployment)."""
    if os.getenv("TESTING", "").lower() in ("1", "true", "yes"):
        return os.getenv("MATHPILOT_PUBLIC_DEMO", "").lower() in ("1", "true", "yes")
    return os.getenv("MATHPILOT_PUBLIC_DEMO", "true").lower() in ("1", "true", "yes")


def default_demo_user(db: Session) -> User | None:
    """The default public demo user (Dr. Sarah Wilson, or the first instructor)."""
    try:
        return (
            db.query(User).filter(User.name == "Dr. Sarah Wilson").first()
            or db.query(User).filter(User.role == "instructor").order_by(User.id).first()
            or db.query(User).first()
        )
    except Exception:
        return None


def viewer(request: Request, db: Session) -> User | None:
    """The signed-in roster user, or the default public demo instructor.

    Anyone visiting the deployed site without an account or cookie is given
    public demo access as Dr. Sarah Wilson so they can explore all features
    (AI tutor, lesson builder, assignment generation, analytics) without
    manual sign-in. Users can still switch accounts or explicitly sign out.
    """
    cookie_val = request.cookies.get(COOKIE)
    if cookie_val is not None:
        uid = resolve(cookie_val)
        if uid == 0:
            # Explicit sign-out
            return None
        if uid is not None:
            return db.query(User).filter(User.id == uid).first()
        # Invalid / tampered cookie provided: reject rather than defaulting
        return None
    # Public Demo Access: No cookie set -> default to Dr. Sarah Wilson
    if is_public_demo():
        return default_demo_user(db)
    return None


def viewer_or_demo_student(request: Request, db) -> User:
    """Identity for a student-facing page.

    With no cookie the demo roster's default public demo user is used.
    An instructor viewing a student page sees that instructor's own work.
    """
    who = viewer(request, db)
    if who is not None:
        return who
    return db.query(User).filter(User.role == "student").order_by(
        User.id).first()


def require_instructor(request: Request,
                       db: Session = Depends(get_db)) -> User:
    """FastAPI dependency: instructor role or 403."""
    who = viewer(request, db)
    if who is None:
        raise HTTPException(403, "Sign in as an instructor to view class "
                                 "insights — use View as in the header.")
    if who.role != "instructor":
        raise HTTPException(403, f"Signed in as {who.name} (student). Class "
                                 "insights are instructor-only.")
    return who


def set_viewer(response, user: User):
    response.set_cookie(
        key=COOKIE, value=issue(user.id), max_age=MAX_AGE, httponly=True,
        samesite="lax", path="/",
        # Off by default so the demo works on http://localhost; turn on with
        # MATHPILOT_COOKIE_SECURE once this is served over https.
        secure=os.getenv("MATHPILOT_COOKIE_SECURE", "").lower() in ("1", "true", "yes"))
    return response


def set_signed_out(response):
    """Set cookie indicating explicit signed-out state."""
    response.set_cookie(
        key=COOKIE, value=issue(0), max_age=MAX_AGE, httponly=True,
        samesite="lax", path="/",
        secure=os.getenv("MATHPILOT_COOKIE_SECURE", "").lower() in ("1", "true", "yes"))
    return response


def clear_viewer(response):
    response.delete_cookie(key=COOKIE, path="/")
    return response


def initials(name: str) -> str:
    parts = [p for p in (name or "").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def add_viewer_context(templates: Jinja2Templates):
    """Expose the current viewer and the roster to every template.

    Jinja cannot open a database session, so the lookups happen here and the
    template only renders the result. `demo` is exposed here rather than
    route by route because the sidebar attribution needs it on every page.
    """
    from . import config
    from .db import SessionLocal

    def _context(request: Request) -> dict:
        db = SessionLocal()
        try:
            who = viewer(request, db)
            roster = db.query(User).order_by(User.role, User.name).all()
            return {
                "viewer": who,
                "viewer_initials": initials(who.name) if who else "—",
                "roster": roster,
                "demo": not config.ai_configured(),
            }
        finally:
            db.close()

    templates.context_processors.append(_context)
    return templates
