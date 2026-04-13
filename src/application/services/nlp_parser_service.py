"""NLP parser service for parsing natural language alerts."""
import re
from typing import Dict, Any
from src.domain.interfaces.llm_provider import LLMProvider
from src.shared.logging import get_logger

logger = get_logger(__name__)


class NLPParserService:
    """Service for parsing natural language alert requests."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize NLP parser service.
        
        Args:
            llm_provider: LLM provider for parsing natural language
        """
        self.llm_provider = llm_provider
        logger.info("Initialized NLPParserService")

    async def parse_alert_intent(self, natural_language_input: str) -> Dict[str, Any]:
        """
        Parse natural language alert request and extract structured information.
        
        Args:
            natural_language_input: Natural language alert request
            
        Returns:
            Dictionary with parsed alert information
        """
        logger.info(f"Parsing alert intent: {natural_language_input[:100]}...")
        
        prompt = f"""Parse the following alert request and extract structured information:

"{natural_language_input}"

Extract:
1. Stock ticker symbol
2. Condition (price/volume/technical indicator)
3. Threshold value
4. Timeframe
5. Alert type

Respond in JSON format with keys: ticker, condition, threshold, timeframe, alert_type"""

        try:
            response = await self.llm_provider.generate(prompt)
            logger.debug("Received parsing response")
            
            # Simple parsing (in production, use structured output from LLM)
            ticker_match = re.search(r'\b([A-Z]{2,5})\b', natural_language_input.upper())
            ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
            
            threshold_match = re.search(r'(\d+(?:\.\d+)?)\s*%', natural_language_input)
            threshold = float(threshold_match.group(1)) if threshold_match else 5.0
            
            condition = "price"
            input_lower = natural_language_input.lower()
            if "volume" in input_lower:
                condition = "volume"
            elif "rsi" in input_lower or "macd" in input_lower:
                condition = "technical_indicator"
            
            timeframe = "this week"
            if "today" in input_lower:
                timeframe = "today"
            elif "month" in input_lower:
                timeframe = "this month"
            
            alert_type = (
                "price" if condition == "price"
                else "volume" if condition == "volume"
                else "technical_indicator"
            )
            
            logger.info(f"Parsed alert: {ticker} {condition} {threshold}% {timeframe}")

            return {
                "ticker": ticker,
                "condition": condition,
                "threshold": threshold,
                "timeframe": timeframe,
                "alert_type": alert_type
            }
        except Exception as e:
            logger.error(f"Error parsing alert intent: {str(e)}")
            raise
