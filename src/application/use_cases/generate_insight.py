"""Use case for generating trading insights."""
from typing import Dict, Any, Optional
from src.application.services.insight_service import InsightService
from src.shared.logging import get_logger

logger = get_logger(__name__)


class GenerateInsightUseCase:
    """Use case for generating AI-based trading insights."""
    
    def __init__(self, insight_service: InsightService):
        """
        Initialize insight use case.
        
        Args:
            insight_service: Insight service for generating insights
        """
        self.insight_service = insight_service
        logger.info("Initialized GenerateInsightUseCase")

    async def execute(
        self,
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        mode: str = "strict"
    ) -> Dict[str, Any]:
        """
        Execute insight generation use case.
        
        Args:
            symbol: Stock symbol
            technical_data: Technical indicators
            fundamental_data: Fundamental metrics
            sentiment_data: Sentiment analysis
            
        Returns:
            Insight dictionary
        """
        logger.info(f"Executing insight generation for {symbol}")
        return await self.insight_service.generate_insight(
            symbol=symbol,
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            sentiment_data=sentiment_data,
            mode=mode
        )
