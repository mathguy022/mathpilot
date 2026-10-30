"""Vercel serverless entrypoint for MathPilot."""
import sys
from pathlib import Path

# Ensure root workspace directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app as _base_app  # noqa: E402


async def app(scope, receive, send):
    """ASGI wrapper to normalize paths on Vercel."""
    if scope["type"] == "http":
        path = scope.get("path", "")
        for prefix in ("/api/index.py", "/api/index"):
            if path == prefix:
                scope["path"] = "/"
                break
            elif path.startswith(prefix + "/"):
                scope["path"] = path[len(prefix):]
                break
    await _base_app(scope, receive, send)
