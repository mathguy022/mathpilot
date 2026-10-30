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

# Without a configured secret, sessions stay valid only for this process.
_SECRET = (os.getenv(SECRET_ENV) or "").encode() or secrets.token_bytes(32)


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


def viewer(request: Request, db) -> User | None:
    """The signed-in roster user, or None. Role comes from the database."""
    uid = resolve(request.cookies.get(COOKIE))
    if uid is None:
        return None
    return db.query(User).filter(User.id == uid).first()


def viewer_or_demo_student(request: Request, db) -> User:
    """Identity for a student-facing page.

    With no cookie the demo roster's first student is used, which is what the
    app already assumed before any identity existed. An instructor viewing a
    student page sees that instructor's own (empty) work, not someone else's.
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
