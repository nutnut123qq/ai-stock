"""Analyze API routes."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.application.use_cases.analyze_event import AnalyzeEventUseCase
from src.api.dependencies import get_analyze_event_use_case

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request model for event analysis."""
    description: str


class AnalyzeResponse(BaseModel):
    """Response model for analysis."""
    analysis: str
    impact: str


@router.post("/analyze-event", response_model=AnalyzeResponse)
async def analyze_event(
    request: AnalyzeRequest,
    use_case: AnalyzeEventUseCase = Depends(get_analyze_event_use_case)
):
    """
    Analyze corporate event.

    Args:
        request: Analyze request with event description
        use_case: Analyze event use case instance

    Returns:
        Analysis with impact assessment
    """
    result = await use_case.execute(request.description)
    return AnalyzeResponse(
        analysis=result["analysis"],
        impact=result["impact"]
    )
