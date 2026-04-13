"""Sentiment analysis service for events."""
from typing import Dict, Any
from src.domain.interfaces.llm_provider import LLMProvider
from src.shared.logging import get_logger

logger = get_logger(__name__)


class SentimentService:
    """Service for analyzing sentiment of corporate events."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize sentiment service.
        
        Args:
            llm_provider: LLM provider for sentiment analysis
        """
        self.llm_provider = llm_provider
        logger.info("Initialized SentimentService")

    async def analyze_event(self, event_description: str) -> Dict[str, Any]:
        """
        Analyze corporate event and assess its impact.
        
        Args:
            event_description: Description of the corporate event
            
        Returns:
            Dictionary with analysis and impact assessment
        """
        logger.info(f"Analyzing event: {event_description[:100]}...")
        
        prompt = f"""Analyze the following corporate event and assess its impact:

{event_description}

Please provide:
1. Detailed analysis of the event
2. Expected impact on stock price (positive/negative/neutral)
"""

        try:
            response = await self.llm_provider.generate(prompt)
            logger.debug("Received sentiment analysis response")
            
            impact = "neutral"
            response_lower = response.lower()
            if "positive" in response_lower or "increase" in response_lower:
                impact = "positive"
            elif "negative" in response_lower or "decrease" in response_lower:
                impact = "negative"
            
            logger.info(f"Event analysis completed with impact: {impact}")

            return {
                "analysis": response,
                "impact": impact
            }
        except Exception as e:
            logger.error(f"Error analyzing event: {str(e)}")
            raise
