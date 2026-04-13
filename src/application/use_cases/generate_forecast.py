"""Use case for generating stock forecasts."""
from typing import Dict, Any, Optional
from src.application.services.forecast_service import ForecastService
from src.shared.logging import get_logger

logger = get_logger(__name__)


class GenerateForecastUseCase:
    """Use case for generating AI-based stock forecasts."""
    
    def __init__(self, forecast_service: ForecastService):
        """
        Initialize forecast use case.
        
        Args:
            forecast_service: Forecast service for generating forecasts
        """
        self.forecast_service = forecast_service
        logger.info("Initialized GenerateForecastUseCase")

    async def execute(
        self,
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        time_horizon: str = "short"
    ) -> Dict[str, Any]:
        """
        Execute forecast generation use case.
        
        Args:
            symbol: Stock symbol
            technical_data: Technical indicators
            fundamental_data: Fundamental metrics
            sentiment_data: Sentiment analysis
            time_horizon: Forecast time period
            
        Returns:
            Forecast dictionary
        """
        logger.info(f"Executing forecast generation for {symbol}")
        return await self.forecast_service.generate_forecast(
            symbol=symbol,
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            sentiment_data=sentiment_data,
            time_horizon=time_horizon
        )
