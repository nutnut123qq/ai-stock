"""Forecast service for generating stock forecasts."""
import json
import re
from typing import Dict, Any, Optional
from src.domain.interfaces.llm_provider import LLMProvider
from src.application.services.prompt_builder import PromptBuilder
from src.shared.utils import (
    extract_trend,
    extract_confidence,
    extract_recommendation,
    extract_list_items
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


class ForecastService:
    """Service for generating AI-based stock forecasts."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize forecast service.
        
        Args:
            llm_provider: LLM provider for generating forecasts
        """
        self.llm_provider = llm_provider
        logger.info("Initialized ForecastService")

    async def generate_forecast(
        self,
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        time_horizon: str = "short"  # short (1-5 days), medium (1-4 weeks), long (1-3 months)
    ) -> Dict[str, Any]:
        """
        Generate AI-based stock forecast using multiple data sources.

        Args:
            symbol: Stock symbol (e.g., VIC, VNM)
            technical_data: Technical indicators (MA, RSI, MACD, etc.)
            fundamental_data: Financial metrics (ROE, ROA, EPS, etc.)
            sentiment_data: News sentiment analysis
            time_horizon: Forecast time period

        Returns:
            Forecast with trend, confidence, price targets, and analysis
        """
        logger.info(f"Generating forecast for {symbol} with time_horizon={time_horizon}")
        
        # Build comprehensive prompt
        prompt = PromptBuilder.build_forecast_prompt(
            symbol=symbol,
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            sentiment_data=sentiment_data,
            time_horizon=time_horizon
        )

        # Generate forecast using LLM provider
        try:
            response = await self.llm_provider.generate(prompt)
            logger.debug(f"Received forecast response for {symbol}")
        except Exception as e:
            logger.error(f"Error generating forecast for {symbol}: {str(e)}")
            raise

        # Parse and structure the response
        forecast = self._parse_forecast_response(response, symbol, time_horizon)
        logger.info(f"Successfully generated forecast for {symbol}: {forecast.get('trend')} with {forecast.get('confidence')} confidence")

        return forecast

    def _parse_forecast_response(
        self,
        response: str,
        symbol: str,
        time_horizon: str
    ) -> Dict[str, Any]:
        """Parse AI response into structured forecast."""
        logger.debug(f"Parsing forecast response for {symbol}")

        # Extract trend
        trend = extract_trend(response)

        # Extract confidence level
        confidence, confidence_score = extract_confidence(response)

        # Extract recommendation
        recommendation = extract_recommendation(response)

        # Extract key drivers and risks
        key_drivers = extract_list_items(response, ["yếu tố", "driver", "lý do"])
        risks = extract_list_items(response, ["rủi ro", "risk", "nguy cơ"])

        return {
            "symbol": symbol,
            "trend": trend,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "time_horizon": time_horizon,
            "recommendation": recommendation,
            "key_drivers": key_drivers[:3] if key_drivers else ["Phân tích kỹ thuật", "Tâm lý thị trường"],
            "risks": risks[:3] if risks else ["Biến động thị trường", "Rủi ro vĩ mô"],
            "analysis": response,
            "generated_at": None  # Will be set by API
        }
