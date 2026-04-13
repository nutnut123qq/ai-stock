"""Insights API routes."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import hashlib
from src.application.use_cases.generate_insight import GenerateInsightUseCase
from src.api.dependencies import get_generate_insight_use_case
from src.shared.config import get_settings

router = APIRouter()
settings = get_settings()


class InsightRequest(BaseModel):
    """Request model for insight generation."""
    symbol: str
    technical_data: Optional[dict] = None
    fundamental_data: Optional[dict] = None
    sentiment_data: Optional[dict] = None


class InsightResponse(BaseModel):
    """Response model for insight."""
    symbol: str
    type: str  # Buy, Sell, Hold
    title: str
    description: str
    confidence: int  # 0-100
    reasoning: List[str]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    evidence: List[str] = []
    metadata: Dict[str, str] = {}
    generated_at: str


class BatchInsightRequest(BaseModel):
    """Request model for batch insight generation."""
    symbols: List[str]
    technical_data: Optional[dict] = None  # Same for all symbols
    fundamental_data: Optional[dict] = None
    sentiment_data: Optional[dict] = None


class BatchInsightResponse(BaseModel):
    """Response model for batch insights."""
    insights: List[InsightResponse]


@router.post("/generate", response_model=InsightResponse)
async def generate_insight(
    request: InsightRequest,
    use_case: GenerateInsightUseCase = Depends(get_generate_insight_use_case)
):
    """
    Generate AI-based trading insight (Buy/Sell/Hold signal) for a stock.

    Args:
        request: Insight request with symbol and optional data
        use_case: Insight use case instance

    Returns:
        Trading insight with type, confidence, and reasoning
    """
    hash_bucket = int(hashlib.sha256(request.symbol.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    canary_enabled = hash_bucket <= settings.insight_canary_ratio
    serving_mode = "strict" if canary_enabled else "legacy"

    result = await use_case.execute(
        symbol=request.symbol,
        technical_data=request.technical_data,
        fundamental_data=request.fundamental_data,
        sentiment_data=request.sentiment_data,
        mode=serving_mode
    )

    if settings.insight_shadow_mode:
        shadow_mode = "legacy" if serving_mode == "strict" else "strict"
        shadow = await use_case.execute(
            symbol=request.symbol,
            technical_data=request.technical_data,
            fundamental_data=request.fundamental_data,
            sentiment_data=request.sentiment_data,
            mode=shadow_mode
        )
        result.setdefault("metadata", {})
        result["metadata"]["shadow_compare_type"] = f"{result.get('type')}|{shadow.get('type')}"
        result["metadata"]["shadow_compare_confidence"] = f"{result.get('confidence')}|{shadow.get('confidence')}"
        result["metadata"]["shadow_mode"] = "enabled"

    result["generated_at"] = datetime.now().isoformat()
    return InsightResponse(**result)


@router.post("/generate/batch", response_model=BatchInsightResponse)
async def generate_batch_insights(
    request: BatchInsightRequest,
    use_case: GenerateInsightUseCase = Depends(get_generate_insight_use_case)
):
    """
    Generate insights for multiple symbols.

    Args:
        request: Batch insight request with list of symbols
        use_case: Insight use case instance

    Returns:
        List of insights for all symbols
    """
    from src.shared.logging import get_logger
    
    logger = get_logger(__name__)
    insights = []
    for symbol in request.symbols:
        try:
            result = await use_case.execute(
                symbol=symbol,
                technical_data=request.technical_data,
                fundamental_data=request.fundamental_data,
                sentiment_data=request.sentiment_data
            )
            result["generated_at"] = datetime.now().isoformat()
            insights.append(InsightResponse(**result))
        except Exception as e:
            # Log error for resilience but continue with other symbols
            logger.warning(
                f"Failed to generate insight for symbol {symbol}",
                extra={"symbol": symbol, "error": str(e)}
            )
            insights.append(InsightResponse(
                symbol=symbol,
                type="Hold",
                title="Error",
                description="Không thể phân tích do lỗi hệ thống",
                confidence=0,
                reasoning=["Lỗi khi phân tích"],
                generated_at=datetime.now().isoformat()
            ))

    return BatchInsightResponse(insights=insights)
