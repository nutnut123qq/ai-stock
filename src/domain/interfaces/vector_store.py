from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class VectorStore(ABC):
    @abstractmethod
    async def search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with optional filters.
        
        Args:
            query_text: Query text for embedding and search
            top_k: Maximum number of results
            filters: Optional filters (document_id, source, symbol)
            
        Returns:
            List of result dictionaries with metadata
        """
        raise NotImplementedError("Subclass must implement search()")

    @abstractmethod
    async def upsert(self, vectors: List[Dict]):
        """Upsert vectors into the collection."""
        raise NotImplementedError("Subclass must implement upsert()")

    @abstractmethod
    async def ensure_collection(self, vector_size: Optional[int] = None) -> None:
        """Ensure the vector collection exists with correct params."""
        raise NotImplementedError("Subclass must implement ensure_collection()")

    @abstractmethod
    async def upsert_chunks(
        self,
        document_id: str,
        source: str,
        payloads: List[Dict[str, Any]],
        vectors: List[List[float]]
    ) -> None:
        """Upsert chunk payloads + vectors into the collection."""
        raise NotImplementedError("Subclass must implement upsert_chunks()")

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        """Delete all points for a document, returning deleted count."""
        raise NotImplementedError("Subclass must implement delete_document()")

