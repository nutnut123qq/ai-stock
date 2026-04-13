"""Dependency injection for API routes."""
from functools import lru_cache
from src.infrastructure.llm.gemini_client import GeminiClient
from src.domain.interfaces.llm_provider import LLMProvider
from src.infrastructure.vector_store.qdrant_client import QdrantClient
from src.infrastructure.vector_store.embedding_service import EmbeddingService
from src.application.services.forecast_service import ForecastService
from src.application.services.insight_service import InsightService
from src.application.services.qa_service import QAService
from src.application.services.summarization_service import SummarizationService
from src.application.services.sentiment_service import SentimentService
from src.application.services.nlp_parser_service import NLPParserService
from src.application.services.stock_data_service import StockDataService
from src.application.services.rag_ingest_service import RagIngestService
from src.application.use_cases.summarize_news import SummarizeNewsUseCase
from src.application.use_cases.answer_question import AnswerQuestionUseCase
from src.application.use_cases.generate_forecast import GenerateForecastUseCase
from src.application.use_cases.generate_insight import GenerateInsightUseCase
from src.application.use_cases.analyze_event import AnalyzeEventUseCase
from src.application.use_cases.parse_alert import ParseAlertUseCase
from src.shared.logging import get_logger
from src.shared.config import get_settings

logger = get_logger(__name__)


# Infrastructure dependencies (singletons)
@lru_cache()
def get_llm_provider() -> LLMProvider:
    """Get LLM provider singleton. Uses Gemini if GEMINI_API_KEY is set, otherwise falls back to Blackbox."""
    logger.debug("Creating LLM provider instance")
    settings = get_settings()
    
    # Prefer Gemini if API key is available
    if settings.gemini_api_key:
        logger.info("Using Gemini as LLM provider")
        return GeminiClient()
    if settings.openrouter_api_key:
        from src.infrastructure.llm.openrouter_client import OpenRouterClient

        logger.info("Using OpenRouter as LLM provider")
        return OpenRouterClient(model_name=settings.openrouter_model)
    # Fallback to Blackbox (will raise error if key not set)
    from src.infrastructure.llm.blackbox_client import BlackboxClient

    logger.info("Using Blackbox as LLM provider")
    return BlackboxClient(model_name=settings.blackbox_model)


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


# Application services
def get_forecast_service() -> ForecastService:
    """Get forecast service instance."""
    return ForecastService(get_llm_provider())


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


def get_sentiment_service() -> SentimentService:
    """Get sentiment service instance."""
    return SentimentService(get_llm_provider())


def get_nlp_parser_service() -> NLPParserService:
    """Get NLP parser service instance."""
    return NLPParserService(get_llm_provider())


def get_stock_data_service() -> StockDataService:
    """Get stock data service instance."""
    return StockDataService()


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


def get_analyze_event_use_case() -> AnalyzeEventUseCase:
    """Get analyze event use case instance."""
    return AnalyzeEventUseCase(get_sentiment_service())


def get_parse_alert_use_case() -> ParseAlertUseCase:
    """Get parse alert use case instance."""
    return ParseAlertUseCase(get_nlp_parser_service())
