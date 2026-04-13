"""Summarize API routes."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.application.use_cases.summarize_news import SummarizeNewsUseCase
from src.api.dependencies import get_summarize_news_use_case

router = APIRouter()


class SummarizeRequest(BaseModel):
    """Request model for news summarization."""
    content: str


class SummarizeResponse(BaseModel):
    """Response model for summary."""
    summary: str
    sentiment: str
    impact_assessment: str


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_news(
    request: SummarizeRequest,
    use_case: SummarizeNewsUseCase = Depends(get_summarize_news_use_case)
):
    """
    Summarize news article.

    Args:
        request: Summarize request with content
        use_case: Summarize news use case instance

    Returns:
        Summary with sentiment and impact assessment
    """
    result = await use_case.execute(request.content)
    return SummarizeResponse(
        summary=result["summary"],
        sentiment=result["sentiment"],
        impact_assessment=result["impact_assessment"]
    )
