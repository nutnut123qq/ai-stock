"""RAG ingest service for chunking and embedding documents into Qdrant."""
import re
from typing import Dict, Any, List, Optional
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Chunking defaults (character-based)
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


class RagIngestService:
    """Service for ingesting documents into RAG vector store."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider
    ):
        """
        Initialize RAG ingest service.
        
        Args:
            vector_store: Vector store for storing embeddings
            embedding_provider: Provider for generating embeddings
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        logger.info("Initialized RagIngestService")
    
    async def ingest(
        self,
        document_id: str,
        source: str,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document by chunking, embedding, and upserting to vector store.
        
        Args:
            document_id: Unique document identifier (string)
            source: Source type (e.g., "analysis_report")
            text: Full document text
            metadata: Document metadata (symbol, title, sourceUrl, etc.)
            
        Returns:
            Ingest result with stats
        """
        logger.info(f"Starting ingest for document {document_id}, source={source}")
        
        if not text or not text.strip():
            logger.warning(f"Empty text for document {document_id}, skipping ingestion")
            return {
                "chunksUpserted": 0,
                "documentId": document_id,
                "collection": getattr(self.vector_store, "collection_name", "stock_documents"),
                "status": "ok"
            }

        # Extract metadata fields (handle both camelCase and snake_case)
        symbol = (metadata.get("symbol") or "").strip()
        title = (metadata.get("title") or "Unknown").strip()
        source_url = metadata.get("sourceUrl") or metadata.get("source_url")
        section = (metadata.get("section") or "").strip()

        # Resolve chunking params
        resolved_chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        resolved_chunk_overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
        if resolved_chunk_size <= 0:
            resolved_chunk_size = DEFAULT_CHUNK_SIZE
        if resolved_chunk_overlap < 0:
            resolved_chunk_overlap = 0
        if resolved_chunk_overlap >= resolved_chunk_size:
            resolved_chunk_overlap = max(0, resolved_chunk_size - 1)

        chunks = self._chunk_text(text, resolved_chunk_size, resolved_chunk_overlap)
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")

        payloads: List[Dict[str, Any]] = []
        vectors: List[List[float]] = []

        for chunk_index, chunk_text in enumerate(chunks):
            embedding = await self.embedding_provider.generate_embedding(chunk_text)
            chunk_id = f"{document_id}:{chunk_index}"
            payloads.append({
                "documentId": document_id,
                "source": source,
                "sourceUrl": source_url,
                "title": title,
                "section": section,
                "symbol": symbol,
                "chunkId": chunk_id,
                "text": chunk_text
            })
            vectors.append(embedding)

        await self.vector_store.upsert_chunks(
            document_id=document_id,
            source=source,
            payloads=payloads,
            vectors=vectors
        )

        logger.info(f"Successfully upserted {len(payloads)} chunks for document {document_id}")
        
        collection_name = getattr(self.vector_store, "collection_name", "stock_documents")
        return {
            "chunksUpserted": len(payloads),
            "documentId": document_id,
            "collection": collection_name,
            "status": "ok"
        }

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document and return status with deleted count."""
        deleted = await self.vector_store.delete_document(document_id)
        return {
            "documentId": document_id,
            "deleted": deleted,
            "status": "ok"
        }

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk text with paragraph preference, fallback to fixed size."""
        normalized = re.sub(r"\r\n", "\n", text).strip()
        if not normalized:
            return []

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
        if not paragraphs:
            return []

        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                if chunk_overlap > 0:
                    overlap_text = current[-chunk_overlap:]
                    current = f"{overlap_text}\n\n{paragraph}"
                else:
                    current = paragraph
            else:
                current = paragraph

            if len(current) > chunk_size:
                chunks.extend(self._hard_split(current, chunk_size, chunk_overlap))
                current = ""

        if current:
            chunks.append(current)

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _hard_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split long text into fixed-size overlapping chunks."""
        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            next_start = end - chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks
