"""Management/monitoring endpoints for the AI service."""
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_settings
from src.infrastructure.cache.redis_cache import RedisCacheService
from src.infrastructure.queue.analyze_queue import get_redis_connection, get_analyze_queue

from src.shared.config import Settings
from src.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/manage", tags=["management"])

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def validate_api_key(x_internal_api_key: Optional[str] = Header(None)):
    settings = get_settings()
    expected_key = getattr(settings, "internal_api_key", None) or os.getenv("INTERNAL_API_KEY")
    if not expected_key:
        logger.error("INTERNAL_API_KEY not configured; rejecting management request")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Management auth not configured")
    if not x_internal_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Internal-Api-Key header")
    if not secrets.compare_digest(x_internal_api_key, expected_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-Internal-Api-Key")
    return True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mgmt_redis() -> Optional[redis.Redis]:
    """Return a Redis client with decode_responses=True for mgmt data."""
    try:
        import redis as _redis
        settings = get_settings()
        return _redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=max(settings.redis_socket_timeout, 5.0),
            health_check_interval=30,
        )
    except Exception as exc:
        logger.warning("Management Redis client init failed: %s", exc)
        return None


def _mask_key(key: Optional[str]) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ProviderItem(BaseModel):
    id: str
    name: str
    model: str
    status: str = "unknown"
    latency_ms: int = 0
    quota_remaining: int = -1
    quota_total: int = -1
    key_masked: str = ""
    priority: int = 99
    note: Optional[str] = None


class PipelineInfo(BaseModel):
    light_mode: bool
    provider: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, str]]
    estimated_llm_calls: int


class RagDocumentItem(BaseModel):
    document_id: str
    source: str
    symbol: str
    chunks: int
    size_bytes: int = 0
    ingested_at: Optional[str] = None


class CacheStats(BaseModel):
    hit_rate_percent: float = 0.0
    memory_used_mb: float = 0.0
    total_keys: int = 0
    connected_clients: int = 0
    db_size: int = 0


class CacheTtlUpdate(BaseModel):
    analyze_ttl: Optional[int] = None
    quote_ttl: Optional[int] = None
    history_ttl: Optional[int] = None
    symbols_ttl: Optional[int] = None


class JobItem(BaseModel):
    id: str
    symbol: str
    status: str
    progress: int = 0
    provider: Optional[str] = None
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ParametersResponse(BaseModel):
    temperature: float
    max_tokens: int
    prompt_version: str
    shadow_mode: bool
    canary_ratio: float
    llm_provider: str
    default_model: str
    light_mode_env: Optional[str] = None


class ParametersUpdate(BaseModel):
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=256, le=8192)
    prompt_version: Optional[str] = None
    shadow_mode: Optional[bool] = None
    canary_ratio: Optional[float] = Field(default=None, ge=0, le=1)


class TraceNode(BaseModel):
    name: str
    status: str
    ms: int
    parallel: bool = False


class TraceItem(BaseModel):
    id: str
    symbol: str
    provider: str
    started_at: str
    total_ms: int
    nodes: List[TraceNode]
    result: Dict[str, Any]


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@router.get("/providers", response_model=List[ProviderItem])
async def get_providers(_: bool = Depends(validate_api_key)):
    """Return configured LLM providers with masked keys."""
    settings = get_settings()
    explicit_provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()

    providers: List[ProviderItem] = []
    configs = [
        ("beeknoee", "Beeknoee", settings.beeknoee_api_key, settings.beeknoee_model),
        ("gemini", "Gemini", settings.gemini_api_key, os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
        ("openrouter", "OpenRouter", settings.openrouter_api_key, settings.openrouter_model),
        ("blackbox", "Blackbox", settings.blackbox_api_key, settings.blackbox_model or "blackboxai/openai/gpt-4-turbo"),
        ("ollama", "Ollama (Local)", "N/A", os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")),
    ]

    for idx, (pid, name, key, model) in enumerate(configs, start=1):
        key_masked = _mask_key(key) if key and key != "N/A" else "N/A"
        configured = bool(key) and key != "N/A"
        note: Optional[str] = None
        if pid == "beeknoee":
            note = "Free tier: 1 concurrent req, cooldown 0.8s"
        if pid == "ollama":
            note = f"{os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')}"

        providers.append(ProviderItem(
            id=pid,
            name=name,
            model=model,
            status="configured" if configured else "not_configured",
            key_masked=key_masked,
            priority=idx,
            note=note,
        ))

    return providers


@router.post("/providers/probe")
async def probe_providers(_: bool = Depends(validate_api_key)):
    """Probe each configured provider for latency and availability."""
    settings = get_settings()
    results: List[Dict[str, Any]] = []

    selector_map = {
        "gemini": (settings.gemini_api_key, lambda: __import__("src.infrastructure.llm.gemini_client", fromlist=["GeminiClient"]).GeminiClient()),
        "openrouter": (settings.openrouter_api_key, lambda: __import__("src.infrastructure.llm.openrouter_client", fromlist=["OpenRouterClient"]).OpenRouterClient()),
        "blackbox": (settings.blackbox_api_key, lambda: __import__("src.infrastructure.llm.blackbox_client", fromlist=["BlackboxClient"]).BlackboxClient()),
        "beeknoee": (settings.beeknoee_api_key, lambda: __import__("src.infrastructure.llm.beeknoee_client", fromlist=["BeeknoeeClient"]).BeeknoeeClient()),
    }

    for pid, (has_key, factory) in selector_map.items():
        if not has_key:
            results.append({"id": pid, "status": "not_configured", "latency_ms": 0, "error": None})
            continue
        start = time.time()
        try:
            client = factory()
            await client.generate("Say 'pong' only.")
            latency = int((time.time() - start) * 1000)
            results.append({"id": pid, "status": "online", "latency_ms": latency, "error": None})
        except Exception as exc:
            latency = int((time.time() - start) * 1000)
            results.append({"id": pid, "status": "offline", "latency_ms": latency, "error": str(exc)})

    return {"probedAt": datetime.now(timezone.utc).isoformat(), "results": results}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@router.get("/pipeline", response_model=PipelineInfo)
async def get_pipeline_info(_: bool = Depends(validate_api_key)):
    from src.langgraph_stock.graph import _light_mode_enabled

    light = _light_mode_enabled()
    explicit = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()

    all_nodes = [
        {"id": "context_loader", "name": "Context Loader", "type": "loader", "enabled": True},
        {"id": "news_analyst", "name": "News Analyst", "type": "analyst", "enabled": True},
        {"id": "tech_analyst", "name": "Tech Analyst", "type": "analyst", "enabled": True},
        {"id": "bull_researcher", "name": "Bull Researcher", "type": "researcher", "enabled": True},
        {"id": "bear_researcher", "name": "Bear Researcher", "type": "researcher", "enabled": True},
        {"id": "research_manager", "name": "Research Manager", "type": "synthesizer", "enabled": True},
        {"id": "trader", "name": "Trader", "type": "decision", "enabled": True},
        {"id": "aggressive_debator", "name": "Aggressive Debator", "type": "debator", "enabled": not light},
        {"id": "neutral_debator", "name": "Neutral Debator", "type": "debator", "enabled": not light},
        {"id": "conservative_debator", "name": "Conservative Debator", "type": "debator", "enabled": not light},
        {"id": "risk_judge", "name": "Risk Judge", "type": "final", "enabled": True},
    ]

    edges = []
    edges.append({"from": "START", "to": "context_loader"})
    if explicit == "beeknoee":
        edges.append({"from": "context_loader", "to": "news_analyst"})
        edges.append({"from": "news_analyst", "to": "tech_analyst"})
        edges.append({"from": "tech_analyst", "to": "bull_researcher"})
    else:
        edges.append({"from": "context_loader", "to": "news_analyst"})
        edges.append({"from": "context_loader", "to": "tech_analyst"})
        edges.append({"from": "news_analyst", "to": "bull_researcher"})
        edges.append({"from": "tech_analyst", "to": "bull_researcher"})

    edges.append({"from": "bull_researcher", "to": "bear_researcher"})
    edges.append({"from": "bear_researcher", "to": "research_manager"})
    edges.append({"from": "research_manager", "to": "trader"})

    if light:
        edges.append({"from": "trader", "to": "risk_judge"})
    else:
        edges.append({"from": "trader", "to": "aggressive_debator"})
        edges.append({"from": "aggressive_debator", "to": "neutral_debator"})
        edges.append({"from": "neutral_debator", "to": "conservative_debator"})
        edges.append({"from": "conservative_debator", "to": "risk_judge"})

    return PipelineInfo(
        light_mode=light,
        provider=explicit,
        nodes=all_nodes,
        edges=edges,
        estimated_llm_calls=7 if light else 10,
    )


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

@router.get("/rag/documents", response_model=List[RagDocumentItem])
async def list_rag_documents(_: bool = Depends(validate_api_key)):
    """List unique documents by aggregating Qdrant point payloads."""
    from src.infrastructure.vector_store.qdrant_client import QdrantClient
    from src.infrastructure.vector_store.embedding_service import EmbeddingService
    settings = get_settings()
    vs = QdrantClient(EmbeddingService())
    coll = settings.qdrant_collection_name or "stock_documents"

    try:
        scroll_result = vs.client.scroll(collection_name=coll, limit=1000, with_payload=True, with_vectors=False)
        points = scroll_result[0] if isinstance(scroll_result, tuple) else getattr(scroll_result, "points", [])
    except Exception as exc:
        logger.error("Qdrant scroll failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc

    docs: Dict[str, Dict[str, Any]] = {}
    for pt in points:
        payload = getattr(pt, "payload", pt) if hasattr(pt, "payload") else pt
        if not isinstance(payload, dict):
            continue
        doc_id = payload.get("documentId") or payload.get("document_id") or "unknown"
        if doc_id not in docs:
            docs[doc_id] = {
                "document_id": doc_id,
                "source": payload.get("source", "unknown"),
                "symbol": payload.get("symbol", "ALL"),
                "chunks": 0,
                "size_bytes": 0,
                "ingested_at": payload.get("ingestedAt") or payload.get("created_at"),
            }
        docs[doc_id]["chunks"] += 1
        text = payload.get("text", "")
        docs[doc_id]["size_bytes"] += len(text.encode("utf-8"))

    return [RagDocumentItem(**d) for d in docs.values()]


@router.delete("/rag/documents/{document_id}")
async def delete_rag_document(document_id: str, _: bool = Depends(validate_api_key)):
    """Delete all chunks belonging to a document from Qdrant."""
    from src.infrastructure.vector_store.qdrant_client import QdrantClient
    from src.infrastructure.vector_store.embedding_service import EmbeddingService
    settings = get_settings()
    vs = QdrantClient(EmbeddingService())
    coll = settings.qdrant_collection_name or "stock_documents"

    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    try:
        # Try both payload key variants
        total_deleted = 0
        for key in ("documentId", "document_id"):
            filt = Filter(must=[FieldCondition(key=key, match=MatchValue(value=document_id))])
            vs.client.delete(collection_name=coll, points_filter=filt)
            total_deleted += 1
    except Exception as exc:
        logger.error("Qdrant delete failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Qdrant delete failed: {exc}") from exc

    return {"document_id": document_id, "status": "deleted"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@router.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats(_: bool = Depends(validate_api_key)):
    r = _get_mgmt_redis()
    if r is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    try:
        info = r.info()
        db_size = r.dbsize()
        memory = info.get("used_memory", 0)
        keyspace = info.get("db" + str(get_settings().redis_db), {})
        keys = keyspace.get("keys", db_size)
        clients = info.get("connected_clients", 0)
        # hit rate from keyspace_hits / (hits + misses)
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0.0
        return CacheStats(
            hit_rate_percent=round(rate, 1),
            memory_used_mb=round(memory / (1024 * 1024), 1),
            total_keys=keys,
            connected_clients=clients,
            db_size=db_size,
        )
    except Exception as exc:
        logger.error("Redis stats failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Redis error: {exc}") from exc


@router.put("/cache/ttl")
async def update_cache_ttl(payload: CacheTtlUpdate, _: bool = Depends(validate_api_key)):
    r = _get_mgmt_redis()
    if r is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    updated = {}
    mapping = {
        "analyze_ttl": "ANALYZE_CACHE_TTL",
        "quote_ttl": "VNSTOCK_CACHE_QUOTE_TTL",
        "history_ttl": "VNSTOCK_CACHE_HISTORY_TTL",
        "symbols_ttl": "VNSTOCK_CACHE_SYMBOLS_TTL",
    }
    for field, env_key in mapping.items():
        val = getattr(payload, field)
        if val is not None:
            r.hset("ai:parameters:cache", env_key, str(val))
            updated[field] = val
    return {"updated": updated, "note": "Restart AI service to apply env overrides"}


# ---------------------------------------------------------------------------
# Jobs (RQ)
# ---------------------------------------------------------------------------

@router.get("/jobs", response_model=List[JobItem])
async def list_jobs(_: bool = Depends(validate_api_key)):
    conn = get_redis_connection()
    try:
        from rq import Queue
        from rq.job import Job
        from rq.worker import Worker
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RQ not installed: {exc}") from exc

    queue = Queue("analyze", connection=conn)
    jobs: List[JobItem] = []

    # Queued
    for job in queue.jobs:
        jobs.append(JobItem(
            id=job.id,
            symbol=job.kwargs.get("symbol", "unknown"),
            status="queued",
            progress=0,
            provider=job.meta.get("provider") if hasattr(job, "meta") else None,
            enqueued_at=job.enqueued_at.isoformat() if job.enqueued_at else None,
        ))

    # Started
    for jid in queue.started_job_registry.get_job_ids():
        job = Job.fetch(jid, connection=conn)
        jobs.append(JobItem(
            id=jid,
            symbol=job.kwargs.get("symbol", "unknown") if job else "unknown",
            status="running",
            progress=job.meta.get("progress", 0) if job and hasattr(job, "meta") else 0,
            started_at=job.started_at.isoformat() if job and job.started_at else None,
        ))

    # Finished
    for jid in queue.finished_job_registry.get_job_ids():
        job = Job.fetch(jid, connection=conn)
        jobs.append(JobItem(
            id=jid,
            symbol=job.kwargs.get("symbol", "unknown") if job else "unknown",
            status="completed",
            progress=100,
            completed_at=job.ended_at.isoformat() if job and job.ended_at else None,
        ))

    # Failed
    for jid in queue.failed_job_registry.get_job_ids():
        job = Job.fetch(jid, connection=conn)
        jobs.append(JobItem(
            id=jid,
            symbol=job.kwargs.get("symbol", "unknown") if job else "unknown",
            status="failed",
            progress=0,
            error=(job.exc_info or "Unknown error") if job else None,
        ))

    return jobs


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, _: bool = Depends(validate_api_key)):
    conn = get_redis_connection()
    try:
        from rq.job import Job
        from rq import Queue
        job = Job.fetch(job_id, connection=conn)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        queue = Queue("analyze", connection=conn)
        queue.enqueue_job(job)
        return {"job_id": job_id, "status": "requeued"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retry failed: {exc}") from exc


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, _: bool = Depends(validate_api_key)):
    conn = get_redis_connection()
    try:
        from rq.job import Job
        job = Job.fetch(job_id, connection=conn)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.cancel()
        job.delete()
        return {"job_id": job_id, "status": "cancelled"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cancel failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@router.get("/parameters", response_model=ParametersResponse)
async def get_parameters(_: bool = Depends(validate_api_key)):
    settings = get_settings()
    r = _get_mgmt_redis()
    overrides = {}
    if r:
        try:
            overrides = r.hgetall("ai:parameters")
        except Exception:
            pass

    return ParametersResponse(
        temperature=float(overrides.get("LLM_TEMPERATURE", settings.llm_temperature)),
        max_tokens=int(overrides.get("LLM_MAX_TOKENS", settings.llm_max_tokens)),
        prompt_version=overrides.get("INSIGHT_PROMPT_VERSION", settings.insight_prompt_version),
        shadow_mode=overrides.get("INSIGHT_SHADOW_MODE", str(settings.insight_shadow_mode)).lower() == "true",
        canary_ratio=float(overrides.get("INSIGHT_CANARY_RATIO", settings.insight_canary_ratio)),
        llm_provider=(os.getenv("LLM_PROVIDER") or "ollama").strip().lower(),
        default_model=settings.default_llm_model,
        light_mode_env=os.getenv("LANGGRAPH_LIGHT_MODE"),
    )


@router.put("/parameters")
async def update_parameters(payload: ParametersUpdate, _: bool = Depends(validate_api_key)):
    r = _get_mgmt_redis()
    if r is None:
        raise HTTPException(status_code=503, detail="Redis not available")

    mapping = {}
    if payload.temperature is not None:
        mapping["LLM_TEMPERATURE"] = str(payload.temperature)
    if payload.max_tokens is not None:
        mapping["LLM_MAX_TOKENS"] = str(payload.max_tokens)
    if payload.prompt_version is not None:
        mapping["INSIGHT_PROMPT_VERSION"] = payload.prompt_version
    if payload.shadow_mode is not None:
        mapping["INSIGHT_SHADOW_MODE"] = str(payload.shadow_mode)
    if payload.canary_ratio is not None:
        mapping["INSIGHT_CANARY_RATIO"] = str(payload.canary_ratio)

    if mapping:
        r.hset("ai:parameters", mapping=mapping)

    return {"updated": list(mapping.keys()), "note": "Restart AI service workers to apply all overrides"}


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

@router.get("/traces", response_model=List[TraceItem])
async def list_traces(limit: int = 20, _: bool = Depends(validate_api_key)):
    r = _get_mgmt_redis()
    if r is None:
        return []
    try:
        import json
        raw = r.lrange("ai:traces", 0, limit - 1)
        return [TraceItem(**json.loads(x)) for x in raw]
    except Exception as exc:
        logger.error("Failed to load traces: %s", exc)
        return []


@router.delete("/traces")
async def clear_traces(_: bool = Depends(validate_api_key)):
    r = _get_mgmt_redis()
    if r is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    r.delete("ai:traces")
    return {"status": "cleared"}
