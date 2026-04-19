"""Unit tests for the Redis cache integration of ``StockDataService``.

These tests use ``fakeredis`` so that no real Redis server is required. The
vnstock upstream is stubbed via a counting spy to assert that cache hits
truly avoid hitting the upstream. The goals:

1. hit/miss semantics for each of the three cached operations
2. TTL expiry ejects the entry (simulated by ``DEL``)
3. fail-open when Redis raises on GET / SETEX
4. negative responses (``NotFoundError`` / ``ServiceUnavailableError``) are
   NOT cached, so they can be retried on the next call
5. ``get_multiple_quotes`` benefits from per-symbol caching
"""
from __future__ import annotations

import sys
import types
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd

# --- Stub optional modules that would otherwise be imported transitively. ---
if "vnstock" not in sys.modules:
    vnstock_module = types.ModuleType("vnstock")
    vnstock_module.Vnstock = object
    vnstock_module.Listing = object
    sys.modules["vnstock"] = vnstock_module

import fakeredis  # noqa: E402  (placed after stubbing)

from src.application.services.stock_data_service import (  # noqa: E402
    StockDataService,
)
from src.infrastructure.cache.redis_cache import RedisCacheService  # noqa: E402
from src.shared.exceptions import NotFoundError, ServiceUnavailableError  # noqa: E402


def _make_ohlcv_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame indexed by ``time`` like vnstock would return."""
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df.pop("time"), name="time")
    return df


def _two_day_sample() -> pd.DataFrame:
    return _make_ohlcv_dataframe(
        [
            {"time": "2026-04-15", "open": 160.0, "high": 162.0, "low": 159.0, "close": 161.0, "volume": 1000},
            {"time": "2026-04-16", "open": 161.0, "high": 165.0, "low": 160.5, "close": 164.0, "volume": 1500},
        ]
    )


def _build_service(
    *,
    cache_client: Optional[fakeredis.FakeRedis] = None,
    cache_enabled: bool = True,
) -> StockDataService:
    """Create a ``StockDataService`` with its heavy constructor deps mocked.

    We do not want the real ``Vnstock()`` / ``Listing()`` constructors (they
    reach out to network on import of sponsor extras). Replace them with
    lightweight stand-ins before instantiation.
    """
    with patch.object(StockDataService, "__init__", autospec=True, return_value=None) as init:
        init.return_value = None
        service = StockDataService.__new__(StockDataService)

    service.listing = types.SimpleNamespace()
    service._client = types.SimpleNamespace()
    service._cache = RedisCacheService(client=cache_client) if cache_client is not None else None
    service._cache_enabled = bool(cache_enabled and cache_client is not None)
    service._quote_ttl = 45
    service._history_ttl = 60 * 60
    service._symbols_ttl = 24 * 60 * 60
    service._max_retries = 0
    service._base_backoff_seconds = 0.0
    return service


class StockDataServiceQuoteCacheTests(unittest.TestCase):
    """Positive path: cache hit after a single upstream fetch."""

    def setUp(self) -> None:
        self.fake_redis = fakeredis.FakeRedis(decode_responses=True)
        self.service = _build_service(cache_client=self.fake_redis)
        self.fetch_calls = 0

        df = _two_day_sample()

        def fake_fetch(**kwargs):
            self.fetch_calls += 1
            return df

        self._patcher = patch.object(
            self.service, "_fetch_history_with_fallback", side_effect=fake_fetch
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_quote_miss_then_hit_within_ttl(self) -> None:
        first = self.service.get_stock_quote("VIC", "KBS")
        second = self.service.get_stock_quote("VIC", "KBS")

        self.assertEqual(first["symbol"], "VIC")
        self.assertEqual(first, second)
        self.assertEqual(self.fetch_calls, 1, "cache hit must not re-call vnstock")

        key = "ai:vnstock:quote:VIC:KBS"
        self.assertIsNotNone(self.fake_redis.get(key))
        # TTL must be > 0 and <= configured TTL
        ttl = self.fake_redis.ttl(key)
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, self.service._quote_ttl)

    def test_quote_refetches_after_ttl_expiry(self) -> None:
        self.service.get_stock_quote("VIC", "KBS")
        # Simulate TTL expiry by removing the key.
        self.fake_redis.delete("ai:vnstock:quote:VIC:KBS")
        self.service.get_stock_quote("VIC", "KBS")

        self.assertEqual(self.fetch_calls, 2)

    def test_get_multiple_quotes_uses_per_symbol_cache(self) -> None:
        # First call: all three symbols MISS (3 upstream hits).
        self.service.get_multiple_quotes(["VIC", "VNM", "FPT"], "KBS")
        self.assertEqual(self.fetch_calls, 3)

        # Second call: all HIT — no new upstream call.
        self.service.get_multiple_quotes(["VIC", "VNM", "FPT"], "KBS")
        self.assertEqual(self.fetch_calls, 3)

        # Invalidate one symbol — only that one re-fetches.
        self.fake_redis.delete("ai:vnstock:quote:VNM:KBS")
        self.service.get_multiple_quotes(["VIC", "VNM", "FPT"], "KBS")
        self.assertEqual(self.fetch_calls, 4)


class StockDataServiceNegativeCacheTests(unittest.TestCase):
    """NotFoundError / ServiceUnavailableError must not be cached."""

    def setUp(self) -> None:
        self.fake_redis = fakeredis.FakeRedis(decode_responses=True)
        self.service = _build_service(cache_client=self.fake_redis)

    def test_notfound_is_not_cached(self) -> None:
        calls = {"n": 0}

        def empty_fetch(**kwargs):
            calls["n"] += 1
            # Empty DataFrame triggers NotFoundError.
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        with patch.object(self.service, "_fetch_history_with_fallback", side_effect=empty_fetch):
            with self.assertRaises(NotFoundError):
                self.service.get_stock_quote("ZZZ", "KBS")
            with self.assertRaises(NotFoundError):
                self.service.get_stock_quote("ZZZ", "KBS")

        self.assertEqual(calls["n"], 2, "NotFound responses must not be cached")
        self.assertIsNone(self.fake_redis.get("ai:vnstock:quote:ZZZ:KBS"))

    def test_service_unavailable_is_not_cached(self) -> None:
        calls = {"n": 0}

        def failing_fetch(**kwargs):
            calls["n"] += 1
            raise RuntimeError("upstream boom")

        with patch.object(self.service, "_fetch_history_with_fallback", side_effect=failing_fetch):
            with self.assertRaises(ServiceUnavailableError):
                self.service.get_stock_quote("VIC", "KBS")
            with self.assertRaises(ServiceUnavailableError):
                self.service.get_stock_quote("VIC", "KBS")

        self.assertEqual(calls["n"], 2)


class StockDataServiceFailOpenTests(unittest.TestCase):
    """If Redis GET/SETEX raises, the service must still return the quote."""

    def _service_with_broken_redis(self) -> StockDataService:
        import redis.exceptions

        class BrokenFakeRedis(fakeredis.FakeRedis):
            def get(self, *args, **kwargs):  # type: ignore[override]
                raise redis.exceptions.RedisError("simulated GET outage")

            def setex(self, *args, **kwargs):  # type: ignore[override]
                raise redis.exceptions.RedisError("simulated SETEX outage")

        broken = BrokenFakeRedis(decode_responses=True)
        return _build_service(cache_client=broken)

    def test_redis_errors_do_not_break_request(self) -> None:
        service = self._service_with_broken_redis()
        with patch.object(
            service, "_fetch_history_with_fallback", return_value=_two_day_sample()
        ):
            result = service.get_stock_quote("VIC", "KBS")

        self.assertEqual(result["symbol"], "VIC")
        self.assertIn("currentPrice", result)


class StockDataServiceHistoryCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_redis = fakeredis.FakeRedis(decode_responses=True)
        self.service = _build_service(cache_client=self.fake_redis)

    def test_history_is_cached_per_parameter_tuple(self) -> None:
        calls = {"n": 0}

        def fetch(**kwargs):
            calls["n"] += 1
            return _two_day_sample()

        with patch.object(self.service, "_fetch_history_with_fallback", side_effect=fetch):
            a1 = self.service.get_historical_data("VIC", "2026-04-01", "2026-04-16", "1D", "KBS")
            a2 = self.service.get_historical_data("VIC", "2026-04-01", "2026-04-16", "1D", "KBS")
            # Different interval → separate key → upstream again.
            b1 = self.service.get_historical_data("VIC", "2026-04-01", "2026-04-16", "1W", "KBS")

        self.assertEqual(a1, a2)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(b1), 2)


class StockDataServiceDisabledCacheTests(unittest.TestCase):
    """With caching disabled (no client) upstream is called every time."""

    def test_cache_disabled_always_fetches(self) -> None:
        service = _build_service(cache_client=None, cache_enabled=False)
        calls = {"n": 0}

        def fetch(**kwargs):
            calls["n"] += 1
            return _two_day_sample()

        with patch.object(service, "_fetch_history_with_fallback", side_effect=fetch):
            service.get_stock_quote("VIC", "KBS")
            service.get_stock_quote("VIC", "KBS")

        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
