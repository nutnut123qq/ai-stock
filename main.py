"""
Unified AI service entrypoint.

Run from the `ai` directory (so package `src` resolves):

    uvicorn main:app --host 0.0.0.0 --port 8000

Or:

    python -m uvicorn main:app --host 0.0.0.0 --port 8000

Forecast queue worker (required for the /api/forecast/langgraph flow):

The LangGraph analyse pipeline runs inside a separate RQ worker process so
slow LLM calls no longer block uvicorn. Start the worker alongside the API
(same CWD = ``ai``)::

    python worker.py

The worker consumes jobs enqueued by ``POST /api/analyze/enqueue`` and stores
results in Redis; the FastAPI ``GET /api/analyze/jobs/{id}`` endpoint reads
those results back. Forecast jobs default to ``LLM_PROVIDER=openrouter``
inside the worker (configured via ``ANALYZE_LLM_PROVIDER``) so AI Insights
can keep using Beeknoee without fighting for the 1/1 concurrent slot.

Typical dev startup order:

    1. redis-server (port 6379)
    2. uvicorn main:app --host 0.0.0.0 --port 8000
    3. python worker.py            # <-- new: RQ worker for /analyze jobs
    4. dotnet run   (in backend-api/src/StockInvestment.Api)
    5. npm run dev  (in frontend)
"""

from src.api.main import app

__all__ = ["app"]
