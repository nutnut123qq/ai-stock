"""Smoke tests for the /api/analyze optimizations.

Covers:
1. ``context_loader`` node fetches news + tech contexts in parallel.
2. ``_llm_health_probe`` returning False makes the endpoint emit a safe
   fallback JSON payload (no graph run).
3. Redis cache hit on a second call avoids rebuilding the graph entirely.

The tests intentionally avoid any network, real Redis, or actual LLM — they
patch module-level seams exposed in ``src.api.langgraph_analyze`` and
``src.langgraph_stock.graph``.
"""
from __future__ import annotations

import asyncio
import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# --- Stub heavy / optional modules so importing the endpoint is cheap. ---
if "vnstock" not in sys.modules:
    vnstock_module = types.ModuleType("vnstock")
    vnstock_module.Vnstock = object
    vnstock_module.Listing = object
    sys.modules["vnstock"] = vnstock_module

import fakeredis  # noqa: E402

from src.api import langgraph_analyze as api_mod  # noqa: E402
from src.api.langgraph_analyze import AnalyzeRequest, analyze_stock  # noqa: E402
from src.infrastructure.cache.redis_cache import RedisCacheService  # noqa: E402
from src.langgraph_stock import graph as graph_mod  # noqa: E402


class _InstantLLM:
    """Minimal LangChain-compatible stub that returns valid JSON instantly."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):  # noqa: D401 — matches LangChain API
        self.calls += 1
        return SimpleNamespace(
            content='{"forecast":"UP","confidence":60,"debate_summary":{}}'
        )


class _FailingLLM:
    """LLM stub that always raises a network-style error on invoke."""

    def invoke(self, messages):  # noqa: D401
        raise ConnectionError(
            "Failed to establish a new connection: [WinError 10061] "
            "actively refused"
        )


class ContextLoaderParallelTests(unittest.TestCase):
    """Verify news + tech retrievers actually run concurrently."""

    def test_retrievers_start_within_100ms_of_each_other(self) -> None:
        news_started: list[float] = []
        tech_started: list[float] = []
        sleep_s = 0.4

        def slow_news(**kwargs):
            news_started.append(time.perf_counter())
            time.sleep(sleep_s)
            return "news ctx"

        def slow_tech(**kwargs):
            tech_started.append(time.perf_counter())
            time.sleep(sleep_s)
            return "tech ctx"

        with patch.object(graph_mod, "_retrieve_news_context", slow_news), \
             patch.object(graph_mod, "_retrieve_tech_summary", slow_tech):
            compiled = graph_mod.build_ta_graph(
                llm=_InstantLLM(), backend_base_url="http://fake"
            )
            result = compiled.invoke(
                {"symbol": "VIC", "news_context": "", "tech_context": ""}
            )

        self.assertEqual(len(news_started), 1)
        self.assertEqual(len(tech_started), 1)
        # If the two retrievers had run sequentially, the second call would
        # only start after the first finished (~`sleep_s` later). Parallel
        # execution puts the start timestamps within a handful of ms.
        delta = abs(news_started[0] - tech_started[0])
        self.assertLess(
            delta,
            sleep_s * 0.5,
            f"Retrievers did not start in parallel (delta={delta:.3f}s)",
        )
        # Sanity: the final payload still has the expected shape.
        self.assertEqual(result.get("symbol"), "VIC")

    def test_preset_contexts_skip_http(self) -> None:
        """When the caller supplies both contexts, no HTTP call should fire."""
        sentinel = object()

        def fail_retrieve(*args, **kwargs):  # pragma: no cover — must not run
            raise AssertionError("retriever should not be called")

        with patch.object(graph_mod, "_retrieve_news_context", fail_retrieve), \
             patch.object(graph_mod, "_retrieve_tech_summary", fail_retrieve):
            compiled = graph_mod.build_ta_graph(
                llm=_InstantLLM(), backend_base_url="http://fake"
            )
            result = compiled.invoke(
                {
                    "symbol": "FPT",
                    "news_context": "preset news",
                    "tech_context": "preset tech",
                }
            )

        self.assertEqual(result.get("symbol"), "FPT")
        self.assertIsNotNone(sentinel)  # keep lint happy


class HealthProbeFallbackTests(unittest.TestCase):
    """When the probe fails, analyze_stock must not execute the graph."""

    def test_probe_failure_returns_safe_fallback_payload(self) -> None:
        def _fail_build_graph(*args, **kwargs):  # pragma: no cover
            raise AssertionError("graph must not be built when probe fails")

        with patch.object(api_mod, "_build_llm", return_value=_FailingLLM()), \
             patch.object(api_mod, "_get_analyze_cache", return_value=None), \
             patch.object(api_mod, "build_ta_graph", _fail_build_graph):
            payload = asyncio.run(analyze_stock(AnalyzeRequest(symbol="VIC")))

        self.assertEqual(payload["symbol"], "VIC")
        self.assertEqual(payload["forecast"], "SIDEWAYS")
        self.assertEqual(payload["confidence"], 50)
        self.assertIn(
            "LLM unavailable",
            payload["debate_summary"]["final_decision"],
        )
        self.assertEqual(payload["news_evidence"], [])
        self.assertEqual(payload["risk_conditions"], [])


class AnalyzeCacheTests(unittest.TestCase):
    """Second call within TTL must be served from Redis without rebuilding."""

    def test_cache_hit_on_second_call_skips_graph(self) -> None:
        fake_client = fakeredis.FakeRedis(decode_responses=True)
        cache = RedisCacheService(client=fake_client)

        graph_calls = {"count": 0}

        class _FakeGraph:
            def invoke(self, state):
                graph_calls["count"] += 1
                return {
                    "symbol": state["symbol"],
                    "forecast": "UP",
                    "confidence": 70,
                    "debate_summary": {
                        "news_agent": "",
                        "tech_agent": "",
                        "final_decision": "graph ran",
                    },
                    "reasoning": "mock",
                    "news_evidence": [],
                    "tech_evidence": {},
                    "risk_conditions": [],
                }

        # Reset any memoized singleton so our patched cache is used.
        api_mod._analyze_cache = None

        with patch.object(api_mod, "_build_llm", return_value=_InstantLLM()), \
             patch.object(api_mod, "build_ta_graph", return_value=_FakeGraph()), \
             patch.object(api_mod, "_get_analyze_cache", return_value=cache):
            first = asyncio.run(analyze_stock(AnalyzeRequest(symbol="VIC")))
            second = asyncio.run(analyze_stock(AnalyzeRequest(symbol="VIC")))

        self.assertEqual(graph_calls["count"], 1)
        self.assertEqual(first["forecast"], "UP")
        self.assertEqual(second, first)

    def test_probe_failure_is_not_cached(self) -> None:
        """Fallback payload must not poison the cache for future healthy calls."""
        fake_client = fakeredis.FakeRedis(decode_responses=True)
        cache = RedisCacheService(client=fake_client)

        api_mod._analyze_cache = None

        # First call: probe fails → fallback returned, cache must stay empty.
        with patch.object(api_mod, "_build_llm", return_value=_FailingLLM()), \
             patch.object(api_mod, "_get_analyze_cache", return_value=cache):
            payload = asyncio.run(analyze_stock(AnalyzeRequest(symbol="VIC")))
        self.assertEqual(payload["forecast"], "SIDEWAYS")

        # Cache should still be empty for this key.
        key = api_mod._cache_key("VIC", "", "")
        self.assertIsNone(cache.get(key))


if __name__ == "__main__":
    unittest.main()
