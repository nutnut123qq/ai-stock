"""RQ-backed job queue for LangGraph /api/analyze.

Two public pieces:

* :func:`get_analyze_queue` — returns the shared ``rq.Queue`` that the FastAPI
  endpoint enqueues to and that the worker process consumes from.
* :func:`run_analyze_job` — the actual work function executed by the worker.
  It forces ``LLM_PROVIDER=openrouter`` (so the forecast pool is fully
  separated from Beeknoee used by AI Insights) and runs the LangGraph
  pipeline, returning a JSON-serialisable dict that RQ stores in Redis.

The queue/result Redis connection reuses ``Settings.redis_*`` so the
existing cache/VNStock infra already validated the credentials.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis

from src.shared.config import get_settings

_log = logging.getLogger(__name__)

ANALYZE_QUEUE_NAME = "analyze"

_redis_conn: Optional[redis.Redis] = None
_queue: Optional[Any] = None


def get_redis_connection() -> redis.Redis:
    """Return a shared redis-py client for RQ (separate from the cache helper).

    RQ needs ``decode_responses=False`` because it stores pickled Python
    objects as binary blobs. We build a dedicated connection with the same
    host/port/db/password as the rest of the app.
    """
    global _redis_conn
    if _redis_conn is not None:
        return _redis_conn

    settings = get_settings()
    # ``socket_timeout=None`` so RQ can set a long timeout for blocking
    # dequeue; a short timeout makes ``redis-py`` raise while the worker idles.
    connect_timeout = max(settings.redis_socket_timeout, 5.0)
    _redis_conn = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        socket_timeout=None,
        socket_connect_timeout=connect_timeout,
        health_check_interval=30,
    )
    return _redis_conn


def get_analyze_queue() -> Any:
    """Return the shared RQ queue (lazy singleton).

    Importing ``rq`` is deferred so the module is cheap to import in tests
    that never touch the queue.
    """
    global _queue
    if _queue is not None:
        return _queue

    from rq import Queue

    conn = get_redis_connection()
    _queue = Queue(ANALYZE_QUEUE_NAME, connection=conn)
    return _queue


def _save_progress(job_id: str, node_name: str, output: Dict[str, Any]) -> None:
    """Append a step to the analysis progress list in Redis (used by worker)."""
    try:
        conn = get_redis_connection()
        key = f"analyze:progress:{job_id}"
        raw = conn.get(key)
        steps: List[Dict[str, Any]] = []
        if raw is not None:
            try:
                steps = json.loads(raw.decode("utf-8")) if isinstance(raw, bytes) else json.loads(raw)
            except Exception:
                steps = []
        steps.append({
            "node": node_name,
            "output": output,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        conn.setex(key, 3600, json.dumps(steps, ensure_ascii=False, default=str))
    except Exception as exc:
        _log.warning("Failed to save progress for job %s node %s: %s", job_id, node_name, exc)


def run_analyze_job(
    symbol: str,
    news_context: str = "",
    tech_context: str = "",
) -> Dict[str, Any]:
    """Execute one LangGraph analyse run — called by the RQ worker.

    Forecasts run on OpenRouter exclusively to free the Beeknoee slot for
    AI insights. We override the env var *inside the worker process* so
    the change cannot leak back into the FastAPI request handlers.
    """
    forecast_provider = (
        os.getenv("ANALYZE_LLM_PROVIDER") or "openrouter"
    ).strip().lower()

    prev_provider = os.environ.get("LLM_PROVIDER")
    os.environ["LLM_PROVIDER"] = forecast_provider

    # Heavier analyse can keep the full debator set; disable light mode
    # unless the operator explicitly opts in.
    prev_light = os.environ.get("LANGGRAPH_LIGHT_MODE")
    if os.getenv("ANALYZE_LIGHT_MODE") is not None:
        os.environ["LANGGRAPH_LIGHT_MODE"] = os.environ["ANALYZE_LIGHT_MODE"]

    try:
        # Imported lazily so the worker does not pull heavy deps at startup
        # when the queue is empty.
        from src.api.langgraph_analyze import _build_llm
        from src.langgraph_stock.graph import build_ta_graph

        llm = _build_llm()
        backend_base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:5000")

        # Resolve job id from RQ so progress can be reported.
        try:
            from rq import get_current_job
            job = get_current_job()
            job_id = job.id if job else None
        except Exception:
            job_id = None

        progress_callback = None
        if job_id:
            def _make_callback(jid: str):
                def callback(node_name: str, output: Dict[str, Any]) -> None:
                    _save_progress(jid, node_name, output)
                return callback
            progress_callback = _make_callback(job_id)

        graph = build_ta_graph(
            llm=llm,
            backend_base_url=backend_base_url,
            progress_callback=progress_callback,
        )

        state = {
            "symbol": symbol,
            "news_context": news_context or "",
            "tech_context": tech_context or "",
        }

        import time
        start_ts = time.time()
        _log.info(
            "run_analyze_job: start symbol=%s provider=%s backend=%s light=%s",
            symbol,
            forecast_provider,
            backend_base_url,
            os.environ.get("LANGGRAPH_LIGHT_MODE", "auto"),
        )
        result = graph.invoke(state)
        elapsed_ms = int((time.time() - start_ts) * 1000)
        _log.info("run_analyze_job: done symbol=%s elapsed=%.1fs", symbol, elapsed_ms / 1000)

        try:
            from src.api.langgraph_analyze import _save_trace
            _save_trace(symbol, forecast_provider, elapsed_ms, result)
        except Exception as exc:
            _log.warning("run_analyze_job: trace save failed: %s", exc)

        return result
    finally:
        if prev_provider is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = prev_provider
        if prev_light is None:
            os.environ.pop("LANGGRAPH_LIGHT_MODE", None)
        else:
            os.environ["LANGGRAPH_LIGHT_MODE"] = prev_light
