"""Alert NLP API routes."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.application.use_cases.parse_alert import ParseAlertUseCase
from src.api.dependencies import get_parse_alert_use_case

router = APIRouter()


class ParseAlertRequest(BaseModel):
    """Request model for alert parsing."""
    input: str


class ParseAlertResponse(BaseModel):
    """Response model for parsed alert."""
    ticker: str
    condition: str
    threshold: float
    timeframe: str
    alert_type: str


@router.post("/parse-alert", response_model=ParseAlertResponse)
async def parse_alert(
    request: ParseAlertRequest,
    use_case: ParseAlertUseCase = Depends(get_parse_alert_use_case)
):
    """
    Parse natural language alert request.

    Args:
        request: Parse alert request with natural language input
        use_case: Parse alert use case instance

    Returns:
        Parsed alert information
    """
    result = await use_case.execute(request.input)
    return ParseAlertResponse(
        ticker=result["ticker"],
        condition=result["condition"],
        threshold=result["threshold"],
        timeframe=result["timeframe"],
        alert_type=result["alert_type"]
    )
