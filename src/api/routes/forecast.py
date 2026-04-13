"""Forecast API routes."""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from src.application.use_cases.generate_forecast import GenerateForecastUseCase
from src.api.dependencies import get_generate_forecast_use_case

router = APIRouter()


class ForecastRequest(BaseModel):
    """Request model for forecast generation."""
    symbol: str
    technical_data: Optional[dict] = None
    fundamental_data: Optional[dict] = None
    sentiment_data: Optional[dict] = None
    time_horizon: str = "short"  # short, medium, long


class ForecastResponse(BaseModel):
    """Response model for forecast."""
    symbol: str
    trend: str  # Up, Down, Sideways
    confidence: str  # High, Medium, Low
    confidence_score: float
    time_horizon: str
    recommendation: str  # Buy, Hold, Sell
    key_drivers: List[str]
    risks: List[str]
    analysis: str
    generated_at: str


@router.post("/generate", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    use_case: GenerateForecastUseCase = Depends(get_generate_forecast_use_case)
):
    """
    Generate AI-based stock forecast.

    Args:
        request: Forecast request with symbol and optional data
        use_case: Forecast use case instance

    Returns:
        Detailed forecast with trend, confidence, and analysis
    """
    result = await use_case.execute(
        symbol=request.symbol,
        technical_data=request.technical_data,
        fundamental_data=request.fundamental_data,
        sentiment_data=request.sentiment_data,
        time_horizon=request.time_horizon
    )

    result["generated_at"] = datetime.now().isoformat()
    return ForecastResponse(**result)


@router.get("/{symbol}", response_model=ForecastResponse)
async def get_forecast(
    symbol: str,
    time_horizon: str = Query("short", description="Time horizon: short, medium, long"),
    use_case: GenerateForecastUseCase = Depends(get_generate_forecast_use_case)
):
    """
    Get AI forecast for a symbol (simplified version without custom data).

    Args:
        symbol: Stock symbol (e.g., VIC, VNM)
        time_horizon: Forecast time period
        use_case: Forecast use case instance

    Returns:
        Forecast based on available data
    """
    # In a real implementation, we would fetch technical/fundamental/sentiment data here
    # For now, we'll use placeholder data
    result = await use_case.execute(
        symbol=symbol,
        technical_data={
            "ma": "Đang trong xu hướng tăng",
            "rsi": "55 (Trung lập)",
            "macd": "Tín hiệu mua yếu",
            "trend": "Tăng nhẹ"
        },
        fundamental_data={
            "roe": "15.5",
            "roa": "8.2",
            "eps": "2500",
            "pe": "12.5"
        },
        sentiment_data={
            "score": "0.65",
            "sentiment": "Tích cực",
            "recent_news": "Tin tức gần đây khá tích cực"
        },
        time_horizon=time_horizon
    )

    result["generated_at"] = datetime.now().isoformat()
    return ForecastResponse(**result)
