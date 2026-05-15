"""Main FastAPI application."""
import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api import langgraph_analyze
from src.api.routes import summarize, forecast, qa, stock_data, insights, rag, financial_yf, management
from src.shared.config import get_settings
from src.shared.exceptions import (
    AIServiceException,
    LLMProviderError,
    LLMQuotaExceededError,
    VectorStoreError,
    EmbeddingServiceError,
    ValidationError,
    ServiceUnavailableError,
    NotFoundError
)
from src.shared.logging import get_logger, set_request_id, get_request_id
import uuid

settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    """Add request ID and track request metadata for logging."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    logger.info(
        "Request started",
        extra={
            "request_id": request_id,
            "route": request.url.path,
            "method": request.method,
        }
    )

    response = await call_next(request)

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Log structured request metadata
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "route": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2)
        }
    )

    response.headers["X-Request-ID"] = request_id
    return response


# Global exception handlers
@app.exception_handler(LLMQuotaExceededError)
async def llm_quota_exceeded_handler(request: Request, exc: LLMQuotaExceededError):
    """Handle LLM quota exceeded errors."""
    request_id = get_request_id()
    logger.error(f"LLM quota exceeded: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "LLM quota exceeded",
            "message": str(exc),
            "type": "LLMQuotaExceededError",
            "request_id": request_id
        }
    )


@app.exception_handler(LLMProviderError)
async def llm_provider_error_handler(request: Request, exc: LLMProviderError):
    """Handle LLM provider errors."""
    request_id = get_request_id()
    logger.error(f"LLM provider error: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": "LLM provider error",
            "message": str(exc),
            "type": "LLMProviderError",
            "request_id": request_id
        }
    )


@app.exception_handler(VectorStoreError)
async def vector_store_error_handler(request: Request, exc: VectorStoreError):
    """Handle vector store errors."""
    request_id = get_request_id()
    logger.error(f"Vector store error: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Vector store error",
            "message": str(exc),
            "type": "VectorStoreError",
            "request_id": request_id
        }
    )


@app.exception_handler(EmbeddingServiceError)
async def embedding_service_error_handler(request: Request, exc: EmbeddingServiceError):
    """Handle embedding service errors."""
    request_id = get_request_id()
    logger.error(f"Embedding service error: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Embedding service error",
            "message": str(exc),
            "type": "EmbeddingServiceError",
            "request_id": request_id
        }
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    request_id = get_request_id()
    logger.warning(f"Validation error: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation error",
            "message": str(exc),
            "type": "ValidationError",
            "request_id": request_id
        }
    )


@app.exception_handler(ServiceUnavailableError)
async def service_unavailable_error_handler(request: Request, exc: ServiceUnavailableError):
    """Handle service unavailable errors."""
    request_id = get_request_id()
    logger.error(f"Service unavailable: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Service unavailable",
            "message": str(exc),
            "type": "ServiceUnavailableError",
            "request_id": request_id
        }
    )


@app.exception_handler(NotFoundError)
async def not_found_error_handler(request: Request, exc: NotFoundError):
    """Handle not found errors."""
    request_id = get_request_id()
    logger.warning(f"Resource not found: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Resource not found",
            "message": str(exc),
            "type": "NotFoundError",
            "request_id": request_id
        }
    )


@app.exception_handler(AIServiceException)
async def ai_service_exception_handler(request: Request, exc: AIServiceException):
    """Handle general AI service exceptions."""
    request_id = get_request_id()
    logger.error(f"AI service error: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "AI service error",
            "message": str(exc),
            "type": "AIServiceException",
            "request_id": request_id
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = get_request_id()
    logger.exception(f"Unexpected error: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": "Exception",
            "request_id": request_id
        }
    )


# Include routers
app.include_router(langgraph_analyze.router, prefix="/api", tags=["langgraph"])
app.include_router(summarize.router, prefix="/api", tags=["summarize"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["forecast"])  # P0 Fix: Correct prefix for forecast
app.include_router(qa.router, prefix="/api", tags=["qa"])
app.include_router(stock_data.router, tags=["stock"])
app.include_router(insights.router, prefix="/api/insights", tags=["insights"])  # P0 Fix: Correct prefix for insights
app.include_router(rag.router, prefix="/api", tags=["rag"]) # RAG ingestion
app.include_router(financial_yf.router, tags=["financial"])
app.include_router(management.router, prefix="/api", tags=["management"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": settings.api_title, "version": settings.api_version}


@app.get("/health")
async def health():
    """Health check endpoint.

    Always returns HTTP 200 (fail-open): Redis outages must not take the AI
    service offline because ``StockDataService`` degrades gracefully without
    cache. The ``redis`` field surfaces the current status so operators can
    still detect a cache miss-storm scenario.
    """
    from src.api.dependencies import get_cache_service

    redis_status = "disabled"
    if settings.vnstock_cache_enabled:
        cache = get_cache_service()
        if cache is None:
            redis_status = "down"
        else:
            try:
                redis_status = "ok" if cache.healthcheck() else "down"
            except Exception:  # noqa: BLE001 — health must never raise
                redis_status = "down"

    return {
        "status": "healthy",
        "service": settings.api_title,
        "redis": redis_status,
        "vnstock_cache_enabled": settings.vnstock_cache_enabled,
    }
