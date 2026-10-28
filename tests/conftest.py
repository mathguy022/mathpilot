"""Point the whole test session at a throwaway database.

Several tutor tests count rows across an entire table, so they wipe it before
measuring. Without this redirect, that wipe deletes the demo sessions the
instructor insights page reads from mathpilot.db.
"""
import os
import pathlib
import shutil
import tempfile

_TMP = tempfile.mkdtemp(prefix="mathpilot-tests-")
os.environ["DATABASE_URL"] = "sqlite:///" + (
    pathlib.Path(_TMP) / "test.db").as_posix()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
