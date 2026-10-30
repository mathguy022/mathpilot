"""Shared helpers for tests that need a signed-in role.

Identity is a demo HMAC cookie (see app/auth.py); these helpers set it on a
TestClient. Tests that care about the role boundary must sign in rather than
rely on the anonymous fallback, because anonymous now means "the demo student".
"""
from __future__ import annotations

from app import auth
from app.db import SessionLocal, User


def _first_user_id(role: str) -> int:
    db = SessionLocal()
    try:
        row = db.query(User).filter(User.role == role).order_by(User.id).first()
        if row is None:
            raise LookupError(f"no seeded {role} in the test database")
        return row.id
    finally:
        db.close()


def sign_in_as(client, role: str):
    """Give this client the cookie of the roster's first `role` user."""
    client.cookies.clear()
    client.cookies.set(auth.COOKIE, auth.issue(_first_user_id(role)))
    return client


def sign_out(client):
    client.cookies.clear()
    return client


def new_teacher_client():
    """A separate client already signed in as the roster instructor.

    Authoring is role-gated, so tests that build a lesson/assignment and then
    check what a student sees need two identities: this one to publish, and a
    plain anonymous client to read as a student.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    return sign_in_as(TestClient(app), "instructor")
