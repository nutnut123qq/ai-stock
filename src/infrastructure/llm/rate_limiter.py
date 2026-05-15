"""Sliding-window rate limiter for LLM providers.

Useful for free-tier APIs (e.g. OpenRouter 20 req/min) where retries alone
are not enough when a batch of requests arrives in a tight loop.
"""

import asyncio
import time
from collections import deque
from typing import Optional

from src.shared.logging import get_logger

logger = get_logger(__name__)


class SlidingWindowRateLimiter:
    """Async rate limiter using a sliding window (requests per minute).

    Because the AI service runs Uvicorn with --workers 1 (per AGENTS.md),
    an in-process limiter is sufficient. If you ever scale to multiple
    workers or replicas, switch to a Redis-backed limiter.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        if max_requests <= 0:
            max_requests = 1
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, label: Optional[str] = None) -> None:
        """Block until a slot is available in the sliding window."""
        async with self._lock:
            now = time.monotonic()
            # Evict timestamps outside the window
            while self._times and now - self._times[0] >= self.window_seconds:
                self._times.popleft()

            if len(self._times) >= self.max_requests:
                # Calculate how long to sleep for the oldest request to expire
                wait = self.window_seconds - (now - self._times[0]) + 0.5
                wait = max(0.5, wait)
                logger.info(
                    "Rate limit window full (%d/%d). Sleeping %.1fs before %s",
                    len(self._times),
                    self.max_requests,
                    wait,
                    label or "next request",
                )
                # Release lock while sleeping so other coroutines can proceed
                # (though with Uvicorn workers=1 and no other await points,
                # this mainly keeps the code pattern clean)
                pass

            if len(self._times) >= self.max_requests:
                # Need to sleep; release lock temporarily
                pass

        # If we are over limit, sleep outside the lock so other tasks don't
        # get starved (in case the limiter is ever shared across providers).
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self.window_seconds:
                    self._times.popleft()
                if len(self._times) < self.max_requests:
                    self._times.append(now)
                    return
                wait = self.window_seconds - (now - self._times[0]) + 0.5
                wait = max(0.5, wait)
            logger.info(
                "Rate limit window full (%d/%d). Sleeping %.1fs before %s",
                len(self._times),
                self.max_requests,
                wait,
                label or "next request",
            )
            await asyncio.sleep(wait)

    def record(self) -> None:
        """Manually record a request timestamp (e.g. after a successful call)."""
        now = time.monotonic()
        while self._times and now - self._times[0] >= self.window_seconds:
            self._times.popleft()
        self._times.append(now)
