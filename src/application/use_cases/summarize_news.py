"""Use case for summarizing news articles."""
from src.application.services.summarization_service import SummarizationService
from src.shared.logging import get_logger

logger = get_logger(__name__)


class SummarizeNewsUseCase:
    """Use case for summarizing news articles."""
    
    def __init__(self, summarization_service: SummarizationService):
        """
        Initialize summarize news use case.
        
        Args:
            summarization_service: Summarization service for summarizing content
        """
        self.summarization_service = summarization_service
        logger.info("Initialized SummarizeNewsUseCase")

    async def execute(self, news_content: str) -> dict:
        """
        Execute news summarization use case.
        
        Args:
            news_content: News article content
            
        Returns:
            Summary dictionary
        """
        logger.info(f"Executing news summarization: {len(news_content)} characters")
        return await self.summarization_service.summarize(news_content)
