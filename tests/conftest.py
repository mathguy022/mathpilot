"""Shared test setup: a throwaway database and role-aware test clients.

Several tutor tests count rows across an entire table, so they wipe it before
measuring. Without this redirect, that wipe deletes the demo sessions the
instructor insights page reads from mathpilot.db.

`teacher_client` exists because authoring is role-gated: a test must sign in
as an instructor to save, approve, publish or assign. Tests for what a student
sees should use the plain `client`, which stays anonymous.
"""
import os
import pathlib
import shutil
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="mathpilot-tests-")
os.environ["DATABASE_URL"] = "sqlite:///" + (
    pathlib.Path(_TMP) / "test.db").as_posix()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture
def teacher_client():
    """A client carrying the roster instructor's signed demo cookie."""
    from tests._identity import new_teacher_client, sign_out

    c = new_teacher_client()
    yield c
    sign_out(c)
