"""QA API routes."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from src.application.services.qa_service import QAService
from src.api.dependencies import get_qa_service

router = APIRouter()


class SourceObjectModel(BaseModel):
    """Source object model matching backend schema."""
    documentId: str = Field(..., description="Document identifier")
    source: str = Field(..., description="Source type")
    sourceUrl: Optional[str] = Field(None, description="Source URL")
    title: str = Field(..., description="Document/section title")
    section: str = Field(..., description="Section heading")
    symbol: str = Field(..., description="Stock symbol")
    chunkId: str = Field(..., description="Chunk identifier (point ID)")
    score: float = Field(..., description="Relevance score")
    textPreview: str = Field(..., description="Text preview (200-400 chars)")


class QARequestV2(BaseModel):
    """Request model for question answering with filters (v2)."""
    question: str = Field(..., description="User question")
    base_context: str = Field("", description="Base context from caller")
    context: Optional[str] = Field(None, description="Backward compatibility alias for base_context")
    top_k: int = Field(6, ge=1, le=20, description="Number of chunks to retrieve")
    document_id: Optional[str] = Field(None, description="Filter by document ID")
    source: Optional[str] = Field(None, description="Filter by source type")
    symbol: Optional[str] = Field(None, description="Filter by symbol")


class QAResponseV2(BaseModel):
    """Response model for answer with source objects."""
    answer: str = Field(..., description="AI-generated answer")
    sources: List[SourceObjectModel] = Field(..., description="Source objects with metadata")


@router.post("/qa", response_model=QAResponseV2)
async def answer_question(
    request: QARequestV2,
    qa_service: QAService = Depends(get_qa_service)
):
    """
    Answer a question using RAG with optional filters.
    
    Supports filtering by document_id, source, and symbol for precise retrieval.
    Backward compatible: accepts "context" as alias for "base_context".

    Args:
        request: QA request with question, context, and filters
        qa_service: QA service instance

    Returns:
        Answer with source objects containing full metadata
    """
    # Backward compatibility: use context if base_context not provided
    base_context = request.base_context or request.context or ""

    try:
        result = await qa_service.answer_question(
            question=request.question,
            base_context=base_context,
            top_k=request.top_k,
            document_id=request.document_id,
            source=request.source,
            symbol=request.symbol
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"QA/LLM failed: {exc!s}",
        ) from exc
    
    # Convert sources to SourceObjectModel objects
    source_hits = [
        SourceObjectModel(
            documentId=src.get("documentId", ""),
            source=src.get("source", ""),
            sourceUrl=src.get("sourceUrl"),
            title=src.get("title", ""),
            section=src.get("section", ""),
            symbol=src.get("symbol", ""),
            chunkId=src.get("chunkId", ""),
            score=src.get("score", 0.0),
            textPreview=src.get("textPreview", "")
        )
        for src in result["sources"]
    ]
    
    return QAResponseV2(
        answer=result["answer"],
        sources=source_hits
    )
