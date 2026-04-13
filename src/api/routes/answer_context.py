"""
Answer with Context endpoint for Analysis Reports Q&A.
V1 Minimal: NO RAG, NO Qdrant.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from src.application.services.answer_context_service import AnswerContextService
from src.api.dependencies import get_llm_provider
from src.domain.interfaces.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

router = APIRouter()


class ContextPart(BaseModel):
    """
    Context part for Q&A (matches C# DTO with snake_case).
    P0 Fix #1: snake_case for Python Pydantic compatibility.
    """
    source_type: str = Field(..., description="Type of source: analysis_report, financial_report, news")
    source_id: str = Field(..., description="Source entity ID")
    title: str = Field(..., description="Title of the source document")
    url: Optional[str] = Field(None, description="Optional URL to the source")
    excerpt: str = Field(..., description="Excerpt from the source")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class AnswerWithContextRequest(BaseModel):
    """Request for answer-with-context endpoint."""
    question: str = Field(..., min_length=3, description="User's question")
    context_parts: List[ContextPart] = Field(..., min_items=1, description="Context parts to answer from")


class AnswerWithContextResponse(BaseModel):
    """
    Response for answer-with-context endpoint.
    P0 Fix #1: snake_case for Python Pydantic compatibility.
    """
    answer: str = Field(..., description="AI-generated answer with inline citations")
    used_sources: List[int] = Field(..., description="0-based indices of context parts used")


@router.post("/answer-with-context", response_model=AnswerWithContextResponse)
async def answer_with_context(
    request: AnswerWithContextRequest,
    llm_provider: LLMProvider = Depends(get_llm_provider)
):
    """
    Answer a question with provided context parts.
    
    This endpoint:
    1. Takes structured context parts (no RAG/Qdrant)
    2. Calls LLM with numbered context
    3. Extracts citations from answer
    4. Returns answer + used source indices
    
    P0 Fixes applied:
    - #13: Strict citation extraction (regex r'\\[(\\d{1,2})\\]')
    - #14: Fallback to [0] if no citations found
    """
    try:
        logger.info(f"Received answer-with-context request with {len(request.context_parts)} context parts")
        
        service = AnswerContextService(llm_provider)
        
        # Convert Pydantic models to dicts for service
        context_parts_dicts = [part.model_dump() for part in request.context_parts]
        
        result = await service.answer_question(
            question=request.question,
            context_parts=context_parts_dicts
        )
        
        return AnswerWithContextResponse(
            answer=result["answer"],
            used_sources=result["used_sources"]
        )
    
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to answer question. Please try again later.")
