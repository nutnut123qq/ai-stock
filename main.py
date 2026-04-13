"""
Unified AI service entrypoint.

Run from the `ai` directory (so package `src` resolves):

    uvicorn main:app --host 0.0.0.0 --port 8000

Or:

    python -m uvicorn main:app --host 0.0.0.0 --port 8000
"""

from src.api.main import app

__all__ = ["app"]
