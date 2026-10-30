"""Instructor Builder page renders for a signed-in teacher.

The mock assign/solve flow these tests used to cover is gone: delivery now
runs entirely through the Publication table, /learn and /myassignments,
covered in tests/test_planc_workflow.py and tests/test_student_workspace.py.
"""
import os

os.environ["DEMO_MODE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from tests._identity import sign_in_as, sign_out  # noqa: E402

client = TestClient(app)


def test_dashboard_renders():
    sign_in_as(client, "instructor")
    r = client.get("/instructor/dashboard")
    sign_out(client)
    assert r.status_code == 200
    assert "Generate Lesson" in r.text
    assert "Create Assignment" in r.text
    assert "Workspace Canvas" in r.text
    assert "Review, Publish & Assign" in r.text


def test_dashboard_is_closed_to_students():
    assert client.get("/instructor/dashboard").status_code == 403
