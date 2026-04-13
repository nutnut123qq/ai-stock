"""Use case for analyzing corporate events."""
from src.application.services.sentiment_service import SentimentService
from src.shared.logging import get_logger

logger = get_logger(__name__)


class AnalyzeEventUseCase:
    """Use case for analyzing corporate events."""
    
    def __init__(self, sentiment_service: SentimentService):
        """
        Initialize analyze event use case.
        
        Args:
            sentiment_service: Sentiment service for analyzing events
        """
        self.sentiment_service = sentiment_service
        logger.info("Initialized AnalyzeEventUseCase")

    async def execute(self, event_description: str) -> dict:
        """
        Execute event analysis use case.
        
        Args:
            event_description: Description of the corporate event
            
        Returns:
            Analysis dictionary
        """
        logger.info(f"Executing event analysis: {event_description[:100]}...")
        return await self.sentiment_service.analyze_event(event_description)
