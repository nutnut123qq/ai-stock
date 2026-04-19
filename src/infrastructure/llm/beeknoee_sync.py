"""Process-wide lock + helper to serialize every Beeknoee API call.

Beeknoee free tier enforces 1/1 concurrent requests per API key. Any overlap
between `BeeknoeeClient.generate()` calls (REST routes) and the LangGraph
node invocations returns HTTP 429 ``concurrent_limit_exceeded``. A single
lock in this standalone module lets both call sites import the same object
without creating a circular import between ``beeknoee_client`` and the graph
module.
"""

from __future__ import annotations

import os
import threading
import time

BEEKNOEE_API_LOCK: threading.Lock = threading.Lock()


def cooldown() -> None:
    """Sleep briefly after a Beeknoee completion.

    Some gateways keep the "in-flight" slot reserved for a short grace period
    after the HTTP response is fully flushed; releasing the lock immediately
    can race the next call and re-trigger 429. The delay is configurable via
    ``BEEKNOEE_LLM_COOLDOWN_SEC`` (default 0.8s).
    """
    try:
        delay = float(os.getenv("BEEKNOEE_LLM_COOLDOWN_SEC", "0.8") or 0.8)
    except ValueError:
        delay = 0.8
    if delay > 0:
        time.sleep(delay)
