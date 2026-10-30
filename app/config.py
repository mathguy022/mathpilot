"""Central application settings. Secrets come from environment only."""
from __future__ import annotations
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env (gitignored) for local runs; real env vars take precedence

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_MATH_MODEL = os.getenv("NEBIUS_MATH_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
NEBIUS_SOCRATIC_MODEL = os.getenv(
    "NEBIUS_SOCRATIC_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
)
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    if os.getenv("VERCEL"):
        DATABASE_URL = "sqlite:////tmp/mathpilot.db"
    else:
        DATABASE_URL = "sqlite:///./mathpilot.db"
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")


def ai_configured() -> bool:
    return bool(NEBIUS_API_KEY) and not DEMO_MODE
