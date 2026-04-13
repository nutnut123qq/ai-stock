"""RAG (Retrieval-Augmented Generation) API routes."""
import os
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
from src.application.services.rag_ingest_service import RagIngestService
from src.api.dependencies import get_rag_ingest_service
from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    document_id: str = Field(..., description="Unique document identifier (string)")
    source: str = Field(..., description="Source type (e.g., 'analysis_report')")
    text: str = Field(..., description="Full document text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    chunk_size: Optional[int] = Field(
        default=None, 
        ge=300, 
        le=4000, 
        description="Chunk size in characters (300-4000)"
    )
    chunk_overlap: Optional[int] = Field(
        default=None, 
        ge=0, 
        le=1000, 
        description="Chunk overlap in characters (0-1000)"
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap_less_than_size(cls, v, info):
        """Ensure chunk_overlap < chunk_size."""
        chunk_size = info.data.get("chunk_size")
        if v is not None and chunk_size is not None and v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


class IngestResponse(BaseModel):
    """Response model for ingestion result."""
    chunksUpserted: int = Field(..., description="Number of chunks upserted")
    documentId: str = Field(..., description="Document identifier")
    collection: str = Field(..., description="Vector store collection name")
    status: str = Field(..., description="Status string")


class DeleteResponse(BaseModel):
    """Response model for delete result."""
    documentId: str = Field(..., description="Document identifier")
    deleted: int = Field(..., description="Number of chunks deleted")
    status: str = Field(..., description="Status string")


def validate_api_key(x_internal_api_key: Optional[str] = Header(None)):
    """
    Validate internal API key header.
    
    Args:
        x_internal_api_key: API key from X-Internal-Api-Key header
        
    Raises:
        HTTPException: 401 if key is missing or invalid
    """
    settings = get_settings()
    expected_key = getattr(settings, "internal_api_key", None) or os.getenv("INTERNAL_API_KEY")
    
    # Validate
    if not expected_key:
        logger.warning("INTERNAL_API_KEY not configured; skipping validation")
        return True
    
    if not x_internal_api_key:
        logger.warning("RAG ingest request missing X-Internal-Api-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Internal-Api-Key header"
        )
    
    if x_internal_api_key != expected_key:
        logger.warning("RAG ingest request with invalid X-Internal-Api-Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Internal-Api-Key"
        )
    
    # Valid key
    return True


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    service: RagIngestService = Depends(get_rag_ingest_service),
    _api_key_valid: bool = Depends(validate_api_key)
):
    """
    Ingest a document into RAG vector store.
    
    This endpoint:
    1. Chunks the document text (paragraph-friendly, fixed-size with overlap)
    2. Generates embeddings for each chunk
    3. Upserts to Qdrant with deterministic point IDs and metadata
    
    Requires X-Internal-Api-Key header for authentication.
    
    Args:
        request: Ingest request with document data
        service: RAG ingest service instance
        _api_key_valid: API key validation result (injected)
        
    Returns:
        Ingest result with statistics
    """
    try:
        logger.info(
            f"Ingesting document {request.document_id}, source={request.source}, "
            f"text_length={len(request.text)}"
        )
        
        result = await service.ingest(
            document_id=request.document_id,
            source=request.source,
            text=request.text,
            metadata=request.metadata,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        logger.info(
            f"Successfully ingested document {request.document_id}: "
            f"{result['chunksUpserted']} chunks"
        )
        
        return IngestResponse(
            chunksUpserted=result["chunksUpserted"],
            documentId=result["documentId"],
            collection=result["collection"],
            status=result["status"]
        )
    
    except Exception as e:
        logger.error(f"Error ingesting document {request.document_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.delete("/doc/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    service: RagIngestService = Depends(get_rag_ingest_service),
    _api_key_valid: bool = Depends(validate_api_key)
):
    """Delete all chunks for a document in vector store."""
    try:
        result = await service.delete_document(document_id)
        return DeleteResponse(
            documentId=result["documentId"],
            deleted=result["deleted"],
            status=result["status"]
        )
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
