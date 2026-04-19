"""RQ worker entrypoint for the LangGraph analyse queue.

Run from the ``ai`` directory (same CWD used for uvicorn so ``src`` resolves)::

    python -m worker

Or equivalently::

    python worker.py

The worker consumes jobs enqueued by ``POST /api/analyze/enqueue`` and writes
results back to Redis so the FastAPI status endpoint can return them.
"""
from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv


def _configure_logging() -> None:
    level = (os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def main() -> int:
    load_dotenv()
    _configure_logging()

    # Import lazily so the module is cheap to import under tests.
    # Windows: no ``os.fork`` → ``SimpleWorker``; no ``SIGALRM`` → ``TimerDeathPenalty``.
    from rq import SimpleWorker, Worker
    from rq.timeouts import TimerDeathPenalty

    from src.infrastructure.queue.analyze_queue import (
        ANALYZE_QUEUE_NAME,
        get_analyze_queue,
        get_redis_connection,
    )

    conn = get_redis_connection()
    queue = get_analyze_queue()

    if sys.platform == "win32":

        class WindowsSimpleWorker(SimpleWorker):
            death_penalty_class = TimerDeathPenalty

        worker_cls = WindowsSimpleWorker
    else:
        worker_cls = Worker
    worker = worker_cls([queue], connection=conn, name=f"analyze-{os.getpid()}")
    logging.getLogger(__name__).info(
        "Starting RQ %s on queue=%s (pid=%s)",
        worker_cls.__name__,
        ANALYZE_QUEUE_NAME,
        os.getpid(),
    )
    # ``with_scheduler=False``: we only need immediate job dispatch.
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
