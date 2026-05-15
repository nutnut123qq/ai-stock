"""Dependency injection for API routes."""
from functools import lru_cache
from typing import Optional

from src.infrastructure.llm.gemini_client import GeminiClient
from src.domain.interfaces.llm_provider import LLMProvider
from src.infrastructure.vector_store.qdrant_client import QdrantClient
from src.infrastructure.vector_store.embedding_service import EmbeddingService
from src.infrastructure.cache.redis_cache import RedisCacheService
from src.application.services.forecast_service import ForecastService
from src.application.services.insight_service import InsightService
from src.application.services.qa_service import QAService
from src.application.services.summarization_service import SummarizationService
from src.application.services.stock_data_service import StockDataService
from src.application.services.rag_ingest_service import RagIngestService
from src.application.use_cases.summarize_news import SummarizeNewsUseCase
from src.application.use_cases.answer_question import AnswerQuestionUseCase
from src.application.use_cases.generate_forecast import GenerateForecastUseCase
from src.application.use_cases.generate_insight import GenerateInsightUseCase
from src.shared.logging import get_logger
from src.shared.config import get_settings

logger = get_logger(__name__)


class LLMProviderSelector:
    """Centralized LLM provider selection logic."""

    def __init__(self, settings):
        self.settings = settings

    def select_provider(self, force_provider: Optional[str] = None) -> LLMProvider:
        """Select LLM provider based on configured priority.

        Args:
            force_provider: Force a specific provider (e.g. 'openrouter' for forecasts)

        Returns:
            LLMProvider instance

        Raises:
            ConfigurationError: If no providers are configured.
        """
        # Define priority order: name, config_check, factory
        priority_order = [
            ("beeknoee", self.settings.beeknoee_api_key, self._create_beeknoee),
            ("gemini", self.settings.gemini_api_key, self._create_gemini),
            ("openrouter", self.settings.openrouter_api_key, self._create_openrouter),
            ("blackbox", True, self._create_blackbox),  # Fallback
        ]

        if force_provider:
            for name, _, factory in priority_order:
                if name == force_provider:
                    logger.info("Using forced LLM provider: %s", force_provider)
                    return factory()
            logger.warning(
                "Forced provider %s not configured; using default priority",
                force_provider,
            )

        # Respect explicit LLM_PROVIDER setting if configured
        explicit = getattr(self.settings, 'llm_provider', None)
        if explicit:
            for name, _, factory in priority_order:
                if name == explicit:
                    logger.info("Using configured LLM provider: %s", explicit)
                    return factory()
            logger.warning("Configured LLM_PROVIDER %s not found in priority list; falling back", explicit)

        for name, key_configured, factory in priority_order:
            if key_configured:
                logger.info("Using %s as LLM provider", name)
                return factory()

        from src.shared.exceptions import ConfigurationError

        raise ConfigurationError("No LLM providers are configured")

    def _create_beeknoee(self) -> LLMProvider:
        from src.infrastructure.llm.beeknoee_client import BeeknoeeClient

        return BeeknoeeClient(model_name=self.settings.beeknoee_model)

    def _create_gemini(self) -> LLMProvider:
        return GeminiClient()

    def _create_openrouter(self) -> LLMProvider:
        from src.infrastructure.llm.openrouter_client import OpenRouterClient

        return OpenRouterClient(model_name=self.settings.openrouter_model)

    def _create_blackbox(self) -> LLMProvider:
        from src.infrastructure.llm.blackbox_client import BlackboxClient

        return BlackboxClient(model_name=self.settings.blackbox_model)


# Infrastructure dependencies (singletons)
@lru_cache()
def get_llm_provider() -> LLMProvider:
    """Get LLM provider singleton using centralized selector."""
    logger.debug("Creating LLM provider instance")
    settings = get_settings()
    selector = LLMProviderSelector(settings)
    return selector.select_provider()


@lru_cache()
def get_forecast_llm_provider() -> LLMProvider:
    """Dedicated provider for /api/forecast/generate.

    Forecast prompts are long and Beeknoee's free tier (glm-4.7-flash) frequently
    hits the Cloudflare 524 origin timeout (~100s upstream). Route only forecast
    traffic through OpenRouter when ``OPENROUTER_API_KEY`` is configured, falling
    back to the global provider otherwise so Beeknoee/Gemini/Blackbox still work.
    """
    settings = get_settings()
    selector = LLMProviderSelector(settings)
    return selector.select_provider(force_provider="openrouter")


@lru_cache()
def get_vector_store() -> QdrantClient:
    """Get vector store singleton."""
    logger.debug("Creating vector store instance")
    return QdrantClient(get_embedding_service())


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton."""
    logger.debug("Creating embedding service instance")
    return EmbeddingService()


@lru_cache()
def get_cache_service() -> Optional[RedisCacheService]:
    """Get Redis cache singleton.

    Returns ``None`` when caching is disabled by config or the Redis server is
    unreachable on startup. Callers must treat ``None`` as a fail-open signal.
    """
    settings = get_settings()
    if not settings.vnstock_cache_enabled:
        logger.info("VNStock cache disabled via VNSTOCK_CACHE_ENABLED=false")
        return None
    try:
        service = RedisCacheService(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
        )
        if service.healthcheck():
            logger.info(
                "Redis cache ready at %s:%s/db%s",
                settings.redis_host,
                settings.redis_port,
                settings.redis_db,
            )
            return service
        logger.warning(
            "Redis PING failed at %s:%s/db%s — falling back to no-cache",
            settings.redis_host,
            settings.redis_port,
            settings.redis_db,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — fail-open on any import/connect error
        logger.warning("Redis cache unavailable, running without cache: %s", exc)
        return None


# Application services
def get_forecast_service() -> ForecastService:
    """Forecast service uses the dedicated forecast provider (OpenRouter preferred)."""
    return ForecastService(get_forecast_llm_provider())


def get_insight_service() -> InsightService:
    """Get insight service instance."""
    return InsightService(get_llm_provider())


def get_qa_service() -> QAService:
    """Get QA service instance."""
    return QAService(
        get_llm_provider(),
        get_vector_store(),
        get_embedding_service()
    )


def get_summarization_service() -> SummarizationService:
    """Get summarization service instance."""
    return SummarizationService(get_llm_provider())


@lru_cache()
def get_stock_data_service() -> StockDataService:
    """Get stock data service singleton (shares Redis cache across requests)."""
    settings = get_settings()
    return StockDataService(
        cache=get_cache_service(),
        cache_enabled=settings.vnstock_cache_enabled,
        quote_ttl=settings.vnstock_cache_quote_ttl,
        history_ttl=settings.vnstock_cache_history_ttl,
        symbols_ttl=settings.vnstock_cache_symbols_ttl,
    )


def get_rag_ingest_service() -> RagIngestService:
    """Get RAG ingest service instance."""
    return RagIngestService(
        get_vector_store(),
        get_embedding_service()
    )


# Use cases
def get_summarize_news_use_case() -> SummarizeNewsUseCase:
    """Get summarize news use case instance."""
    return SummarizeNewsUseCase(get_summarization_service())


def get_answer_question_use_case() -> AnswerQuestionUseCase:
    """Get answer question use case instance."""
    return AnswerQuestionUseCase(get_qa_service())


def get_generate_forecast_use_case() -> GenerateForecastUseCase:
    """Get generate forecast use case instance."""
    return GenerateForecastUseCase(get_forecast_service())


def get_generate_insight_use_case() -> GenerateInsightUseCase:
    """Get generate insight use case instance."""
    return GenerateInsightUseCase(get_insight_service())
