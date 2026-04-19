"""Redis-backed cache service with fail-open semantics.

The Python AI service shares the Redis cluster with the .NET backend but uses
its own logical database (default DB 1) to avoid key collisions. All read /
write operations swallow connection or serialization errors and return a
``None`` / ``False`` sentinel so that callers can transparently fall back to
the upstream data source (vnstock) when cache is unavailable.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import redis
from redis.exceptions import RedisError

from src.shared.logging import get_logger

logger = get_logger(__name__)


class RedisCacheService:
    """Thin wrapper around redis-py with JSON (de)serialization and fail-open IO."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,
        password: Optional[str] = None,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        client: Optional[redis.Redis] = None,
    ) -> None:
        """Create a new cache service.

        Args:
            host/port/db/password: Redis connection parameters.
            socket_timeout: Max seconds blocked on a single read/write.
            socket_connect_timeout: Max seconds blocked on TCP connect.
            client: Optional pre-built client (mainly used by tests that inject
                ``fakeredis.FakeRedis`` so no network is hit).
        """
        if client is not None:
            self._client = client
        else:
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password or None,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                health_check_interval=30,
            )

    def get(self, key: str) -> Optional[Any]:
        """Return the deserialized value or ``None`` on miss / error."""
        try:
            raw = self._client.get(key)
        except RedisError as exc:
            logger.warning("Redis GET failed for %s: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.warning("Redis payload for %s is not valid JSON: %s", key, exc)
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        """Serialize ``value`` as JSON and SETEX it. Returns True on success."""
        try:
            payload = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Redis SET skipped for %s (non-serializable): %s", key, exc)
            return False

        try:
            self._client.setex(name=key, time=max(1, int(ttl_seconds)), value=payload)
            return True
        except RedisError as exc:
            logger.warning("Redis SETEX failed for %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Best-effort delete. Returns False on connection errors."""
        try:
            self._client.delete(key)
            return True
        except RedisError as exc:
            logger.warning("Redis DEL failed for %s: %s", key, exc)
            return False

    def healthcheck(self) -> bool:
        """Return True when Redis answers PING."""
        try:
            return bool(self._client.ping())
        except RedisError as exc:
            logger.warning("Redis PING failed: %s", exc)
            return False
