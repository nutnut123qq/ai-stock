"""POST /api/analyze — LangGraph dashboard forecast (StockAnalyst client on .NET)."""
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.infrastructure.cache import RedisCacheService
from src.langgraph_stock.graph import (
    _BEEKNOEE_LLM_LOCK,
    _llm_failure_message,
    build_ta_graph,
)
from src.shared.config import get_settings

load_dotenv()

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")
_log = logging.getLogger(__name__)

router = APIRouter()

# Lazily-initialized singletons so import cost stays cheap and tests can patch them.
_analyze_cache: Optional[RedisCacheService] = None
_probe_executor: Optional[ThreadPoolExecutor] = None


def _get_analyze_cache() -> Optional[RedisCacheService]:
    """Return a shared RedisCacheService instance (fail-open on construction error)."""
    global _analyze_cache
    if _analyze_cache is not None:
        return _analyze_cache
    if (os.getenv("ANALYZE_CACHE_ENABLED", "true") or "true").strip().lower() != "true":
        return None
    try:
        settings = get_settings()
        _analyze_cache = RedisCacheService(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            socket_timeout=settings.redis_socket_timeout,
        )
    except Exception as exc:  # pragma: no cover — fail-open
        _log.warning("analyze cache disabled (init failed): %s", exc)
        _analyze_cache = None
    return _analyze_cache


def _get_probe_executor() -> ThreadPoolExecutor:
    """Dedicated single-thread pool so probes never queue behind the graph run."""
    global _probe_executor
    if _probe_executor is None:
        _probe_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm-probe"
        )
    return _probe_executor


def _current_model_name() -> str:
    """Return the configured model name for whichever provider is active."""
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    if provider == "blackbox":
        return os.getenv("BLACKBOX_MODEL", "blackboxai/openai/gpt-5.2")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL", "openrouter/free")
    if provider == "beeknoee":
        return os.getenv("BEEKNOEE_MODEL", "glm-4.7-flash")
    return "unknown"


def _bucket_30m(now: Optional[datetime] = None) -> str:
    """Return a key slice that only changes every 30 minutes (UTC)."""
    now = now or datetime.now(timezone.utc)
    half = "0" if now.minute < 30 else "1"
    return now.strftime("%Y%m%d%H") + half


def _ctx_hash(news_context: str, tech_context: str) -> str:
    """Short fingerprint so cache keys change when caller-supplied context changes."""
    if not news_context and not tech_context:
        return "auto"
    raw = (news_context or "") + "|" + (tech_context or "")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _cache_key(symbol: str, news_context: str, tech_context: str) -> str:
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    model = _current_model_name().replace(":", "_")
    return (
        f"analyze:v1:{provider}:{model}:{symbol}:"
        f"{_bucket_30m()}:{_ctx_hash(news_context, tech_context)}"
    )


def _llm_health_probe(llm: Any) -> Tuple[bool, Optional[str]]:
    """Send a 1-token ping to the configured LLM with a strict timeout.

    Returns (True, None) when the provider responds in time, otherwise
    (False, human_readable_hint). Errors are swallowed so the endpoint can
    degrade gracefully instead of bubbling a 500 to the UI.
    """
    timeout_s = float(os.getenv("LLM_HEALTH_PROBE_TIMEOUT", "8") or 8)

    def _invoke() -> None:
        if (os.getenv("LLM_PROVIDER") or "ollama").strip().lower() == "beeknoee":
            with _BEEKNOEE_LLM_LOCK:
                llm.invoke([HumanMessage(content="ping")])
        else:
            llm.invoke([HumanMessage(content="ping")])

    executor = _get_probe_executor()
    future = executor.submit(_invoke)
    try:
        future.result(timeout=timeout_s)
        return True, None
    except FuturesTimeoutError:
        future.cancel()
        return False, f"LLM probe timed out after {timeout_s:.0f}s"
    except Exception as exc:
        return False, _llm_failure_message(exc)


def _save_trace(symbol: str, provider: str, total_ms: int, state: Dict[str, Any]) -> None:
    """Persist a simplified execution trace to Redis for the management UI."""
    try:
        from src.infrastructure.queue.analyze_queue import get_redis_connection
        conn = get_redis_connection()
        trace = {
            "id": f"trace:{symbol}:{int(time.time() * 1000)}",
            "symbol": symbol,
            "provider": provider,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_ms": total_ms,
            "nodes": [
                {"name": "context_loader", "status": "ok", "ms": 0, "parallel": False},
                {"name": "news_analyst", "status": "ok" if state.get("news_agent_text") else "skipped", "ms": 0, "parallel": True},
                {"name": "tech_analyst", "status": "ok" if state.get("tech_agent_text") else "skipped", "ms": 0, "parallel": True},
                {"name": "bull_researcher", "status": "ok" if state.get("bull_arguments_text") else "skipped", "ms": 0, "parallel": False},
                {"name": "bear_researcher", "status": "ok" if state.get("bear_arguments_text") else "skipped", "ms": 0, "parallel": False},
                {"name": "research_manager", "status": "ok" if state.get("research_manager_text") else "skipped", "ms": 0, "parallel": False},
                {"name": "trader", "status": "ok" if state.get("trader_text") else "skipped", "ms": 0, "parallel": False},
                {"name": "aggressive_debator", "status": "ok" if state.get("aggressive_debator_text") else "skipped", "ms": 0, "parallel": False},
                {"name": "neutral_debator", "status": "ok" if state.get("neutral_debator_text") else "skipped", "ms": 0, "parallel": False},
                {"name": "conservative_debator", "status": "ok" if state.get("conservative_debator_text") else "skipped", "ms": 0, "parallel": False},
                {"name": "risk_judge", "status": "ok" if state.get("forecast") else "skipped", "ms": 0, "parallel": False},
            ],
            "result": {
                "forecast": state.get("forecast", "SIDEWAYS"),
                "confidence": state.get("confidence", 50),
                "reasoning": state.get("reasoning", ""),
            },
        }
        conn.lpush("ai:traces", json.dumps(trace))
        conn.ltrim("ai:traces", 0, 199)
    except Exception as exc:
        _log.warning("Failed to save trace: %s", exc)


def _fallback_payload(symbol: str, hint: str) -> Dict[str, Any]:
    """Build the same safe JSON shape the Risk Judge emits when it cannot run."""
    return {
        "symbol": symbol,
        "forecast": "SIDEWAYS",
        "confidence": 50,
        "debate_summary": {
            "news_agent": "",
            "tech_agent": "",
            "final_decision": f"LLM unavailable: {hint}",
        },
        "reasoning": (
            f"Bỏ qua phân tích chi tiết vì LLM không phản hồi ({hint}). "
            "Trả về SIDEWAYS/50 để giữ an toàn; vui lòng thử lại sau."
        ),
        "news_evidence": [],
        "tech_evidence": {},
        "risk_conditions": [],
    }


class AnalyzeRequest(BaseModel):
    symbol: str
    news_context: str = Field(
        default="",
        description="Optional override if backend news fetch fails or is empty",
    )
    tech_context: str = Field(
        default="",
        description="Optional override if backend technical summary fetch fails or is empty",
    )


def _normalize_symbol(raw: str) -> str:
    s = (raw or "").strip().upper()
    if not s:
        raise HTTPException(status_code=400, detail="symbol is required")
    if not _SYMBOL_RE.match(s):
        raise HTTPException(
            status_code=400,
            detail="symbol must be 2–20 alphanumeric characters (e.g. VNM, FPT).",
        )
    return s


def _build_llm():
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY not found in .env (required when LLM_PROVIDER=gemini).",
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
            max_retries=1,
            timeout=20,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "600"))
        num_ctx = os.getenv("OLLAMA_NUM_CTX", "").strip()
        kwargs: dict = {
            "model": model,
            "base_url": base_url,
            "temperature": 0.7,
            "sync_client_kwargs": {"timeout": timeout_s},
            "async_client_kwargs": {"timeout": timeout_s},
        }
        if num_ctx.isdigit():
            kwargs["num_ctx"] = int(num_ctx)
        return ChatOllama(**kwargs)

    if provider == "blackbox":
        api_key = os.getenv("BLACKBOX_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="BLACKBOX_API_KEY not found in .env (required when LLM_PROVIDER=blackbox).",
            )
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("BLACKBOX_BASE_URL", "https://api.blackbox.ai").rstrip("/")
        model = os.getenv("BLACKBOX_MODEL", "blackboxai/openai/gpt-5.2")
        timeout_s = float(os.getenv("BLACKBOX_TIMEOUT", "120"))
        return ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_retries=1,
            request_timeout=timeout_s,
        )

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENROUTER_API_KEY not found in .env (required when LLM_PROVIDER=openrouter).",
            )
        from langchain_openai import ChatOpenAI

        base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        timeout_s = float(os.getenv("OPENROUTER_TIMEOUT", "120"))
        return ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_retries=1,
            request_timeout=timeout_s,
        )

    if provider == "beeknoee":
        api_key = os.getenv("BEEKNOEE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="BEEKNOEE_API_KEY not found in .env (required when LLM_PROVIDER=beeknoee).",
            )
        from langchain_openai import ChatOpenAI

        base_url = os.getenv(
            "BEEKNOEE_BASE_URL", "https://platform.beeknoee.com/api/v1"
        ).rstrip("/")
        model = os.getenv("BEEKNOEE_MODEL", "glm-4.7-flash")
        timeout_s = float(os.getenv("BEEKNOEE_TIMEOUT", "120"))
        return ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            # Beeknoee: avoid SDK retries overlapping a single in-flight slot.
            max_retries=0,
            request_timeout=timeout_s,
        )

    raise HTTPException(
        status_code=500,
        detail=(
            f"Unknown LLM_PROVIDER={provider!r}. "
            "Use 'ollama', 'gemini', 'blackbox', 'openrouter', or 'beeknoee'."
        ),
    )


@router.post("/analyze")
async def analyze_stock(request: AnalyzeRequest):
    symbol = _normalize_symbol(request.symbol)
    news_ctx = request.news_context or ""
    tech_ctx = request.tech_context or ""

    cache = _get_analyze_cache()
    cache_key = _cache_key(symbol, news_ctx, tech_ctx)
    ttl_seconds = int(os.getenv("ANALYZE_CACHE_TTL", "1800") or 1800)

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            _log.debug("analyze cache HIT %s", cache_key)
            return cached

    llm = _build_llm()

    # Beeknoee often allows only one in-flight request per key; a separate health
    # probe plus the graph doubles traffic and a timed-out probe can leave the
    # first HTTP call running, causing 429 on the next call. Optional skip:
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    skip_probe = (os.getenv("BEEKNOEE_SKIP_LLM_PROBE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if provider == "beeknoee" and skip_probe:
        probe_ok, probe_hint = True, None
    else:
        probe_ok, probe_hint = _llm_health_probe(llm)
    if not probe_ok:
        _log.warning("analyze: LLM probe failed for %s (%s)", symbol, probe_hint)
        # Return a well-formed payload but do NOT cache it so the next call can
        # try again once the provider recovers.
        return _fallback_payload(symbol, probe_hint or "unknown error")

    try:
        backend_base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:5197")

        graph = build_ta_graph(llm=llm, backend_base_url=backend_base_url)

        state = {
            "symbol": symbol,
            "news_context": news_ctx,
            "tech_context": tech_ctx,
        }

        start_ts = time.time()
        result = graph.invoke(state)
        elapsed_ms = int((time.time() - start_ts) * 1000)

        _save_trace(symbol, provider, elapsed_ms, result)

        if cache is not None:
            cache.set(cache_key, result, ttl_seconds=ttl_seconds)

        return result

    except HTTPException:
        raise
    except Exception as e:
        _log.error("TA graph analysis failed", exc_info=True, extra={"error_type": type(e).__name__})
        raise HTTPException(
            status_code=500,
            detail="Analysis service temporarily unavailable. Please try again later.",
        ) from e


# ---------------------------------------------------------------------------
# Async job-queue variant (RQ-backed) used by /api/forecast/langgraph on .NET.
# ---------------------------------------------------------------------------


def _cache_key_for_queue(symbol: str, news_ctx: str, tech_ctx: str) -> str:
    """Cache key scoped to the forecast queue provider (OpenRouter)."""
    provider = (
        os.getenv("ANALYZE_LLM_PROVIDER") or "openrouter"
    ).strip().lower()
    model = (
        os.getenv("OPENROUTER_MODEL", "openrouter/free").replace(":", "_")
        if provider == "openrouter"
        else _current_model_name().replace(":", "_")
    )
    return (
        f"analyze:v1:{provider}:{model}:{symbol}:"
        f"{_bucket_30m()}:{_ctx_hash(news_ctx, tech_ctx)}"
    )


@router.post("/analyze/enqueue")
async def enqueue_analyze(request: AnalyzeRequest):
    """Enqueue a LangGraph analyse job and return a jobId for polling.

    Returns HTTP 200 with ``{status:"completed", result}`` on cache hit so
    the caller does not need to poll. Otherwise returns HTTP 202 with
    ``{status:"queued", jobId}``.
    """
    symbol = _normalize_symbol(request.symbol)
    news_ctx = request.news_context or ""
    tech_ctx = request.tech_context or ""

    cache = _get_analyze_cache()
    cache_key = _cache_key_for_queue(symbol, news_ctx, tech_ctx)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            _log.debug("enqueue_analyze cache HIT %s", cache_key)
            return {"status": "completed", "jobId": None, "result": cached}

    try:
        from src.infrastructure.queue.analyze_queue import (
            get_analyze_queue,
            run_analyze_job,
        )

        queue = get_analyze_queue()
        job_timeout = int(os.getenv("ANALYZE_JOB_TIMEOUT", "600") or 600)
        result_ttl = int(os.getenv("ANALYZE_JOB_RESULT_TTL", str(8 * 3600)) or (8 * 3600))
        failure_ttl = int(os.getenv("ANALYZE_JOB_FAILURE_TTL", "3600") or 3600)

        job = queue.enqueue(
            run_analyze_job,
            args=(symbol, news_ctx, tech_ctx),
            job_timeout=job_timeout,
            result_ttl=result_ttl,
            failure_ttl=failure_ttl,
            description=f"analyze:{symbol}",
            meta={"symbol": symbol, "cache_key": cache_key},
        )
    except Exception as exc:
        _log.exception("enqueue_analyze failed for %s", symbol)
        raise HTTPException(
            status_code=503,
            detail="Analyze queue temporarily unavailable. Please try again later.",
        ) from exc

    _log.info("enqueue_analyze: symbol=%s jobId=%s", symbol, job.id)
    return JSONResponse(
        status_code=202,
        content={"status": "queued", "jobId": job.id},
    )


def _map_rq_status(raw_status: Optional[str]) -> str:
    """Normalize RQ job statuses to a small FE-friendly set."""
    status = (raw_status or "").strip().lower()
    if status in ("queued", "deferred", "scheduled"):
        return "queued"
    if status in ("started",):
        return "running"
    if status in ("finished",):
        return "completed"
    if status in ("failed", "stopped", "canceled", "cancelled"):
        return "failed"
    return status or "unknown"


@router.get("/analyze/jobs/{job_id}")
async def get_analyze_job(job_id: str):
    """Return the status (and result when ready) of an enqueued analyse job."""
    try:
        from rq.exceptions import NoSuchJobError
        from rq.job import Job

        from src.infrastructure.queue.analyze_queue import get_redis_connection

        conn = get_redis_connection()
    except Exception as exc:
        _log.exception("get_analyze_job: queue unavailable")
        raise HTTPException(
            status_code=503,
            detail="Analyze queue temporarily unavailable. Please try again later.",
        ) from exc

    try:
        job = Job.fetch(job_id, connection=conn)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or expired")
    except Exception as exc:
        _log.exception("get_analyze_job: fetch failed for %s", job_id)
        raise HTTPException(
            status_code=503,
            detail="Cannot fetch job status. Please try again later.",
        ) from exc

    status = _map_rq_status(job.get_status(refresh=True))

    if status == "completed":
        result = job.result
        # Cache-through so subsequent enqueue calls short-circuit.
        cache_key = (job.meta or {}).get("cache_key") if job.meta else None
        if result is not None and cache_key:
            cache = _get_analyze_cache()
            if cache is not None:
                ttl_seconds = int(os.getenv("ANALYZE_CACHE_TTL", "1800") or 1800)
                try:
                    cache.set(cache_key, result, ttl_seconds=ttl_seconds)
                except Exception as exc:  # pragma: no cover — fail-open
                    _log.debug("analyze result cache set failed: %s", exc)
        return {"status": "completed", "jobId": job_id, "result": result}

    if status == "failed":
        raw_exc = (job.exc_info or "")
        # Last line of traceback is the most useful for UI/logs.
        last_line = ""
        for line in reversed(raw_exc.splitlines()):
            if line.strip():
                last_line = line.strip()
                break
        return {
            "status": "failed",
            "jobId": job_id,
            "error": last_line[:500] if last_line else "unknown failure",
        }

    return JSONResponse(
        status_code=202,
        content={"status": status, "jobId": job_id},
    )
