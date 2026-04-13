"""Qdrant vector store client implementation."""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient as Qdrant
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct
)
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.shared.config import get_settings
from src.shared.exceptions import VectorStoreError
from src.shared.constants import DEFAULT_QDRANT_COLLECTION_NAME, DEFAULT_EMBEDDING_DIMENSION
from src.shared.logging import get_logger

logger = get_logger(__name__)


class QdrantClient(VectorStore):
    """Qdrant vector store client implementing VectorStore interface."""
    
    def __init__(self, embedding_provider: EmbeddingProvider):
        """Initialize Qdrant client."""
        settings = get_settings()
        qdrant_url = settings.qdrant_url
        self.collection_name = settings.qdrant_collection_name or DEFAULT_QDRANT_COLLECTION_NAME
        self.embedding_provider = embedding_provider
        
        try:
            self.client = Qdrant(url=qdrant_url)
            logger.info(f"Connected to Qdrant at {qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise VectorStoreError(f"Failed to connect to Qdrant: {str(e)}") from e

    async def search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection with optional filters.
        
        Args:
            query_text: Query text for embedding and search
            top_k: Maximum number of results to return
            filters: Optional filters (document_id, source, symbol)
            
        Returns:
            List of source objects with metadata and text
            
        Raises:
            VectorStoreError: If search operation fails
        """
        try:
            # Generate embedding from query text
            query_vector = await self.embedding_provider.generate_embedding(query_text)

            # Build filter conditions
            filter_conditions = []
            if filters:
                filter_mapping = {
                    "document_id": "documentId",
                    "source": "source",
                    "symbol": "symbol"
                }
                for filter_key, payload_key in filter_mapping.items():
                    value = filters.get(filter_key)
                    if value is not None:
                        filter_conditions.append(
                            FieldCondition(
                                key=payload_key,
                                match=MatchValue(value=value)
                            )
                        )
            
            # Create filter object if conditions exist
            query_filter = None
            if filter_conditions:
                query_filter = Filter(must=filter_conditions)
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter
            )
            
            # Build source objects with metadata and text
            sources = []
            for hit in results:
                payload = hit.payload or {}

                # Build source object (use safe fallbacks)
                source_obj = {
                    "documentId": payload.get("documentId", ""),
                    "source": payload.get("source", ""),
                    "sourceUrl": payload.get("sourceUrl"),
                    "title": payload.get("title", "Unknown"),
                    "section": payload.get("section", ""),
                    "symbol": payload.get("symbol", ""),
                    "chunkId": payload.get("chunkId", str(hit.id)),  # Point ID
                    "score": float(hit.score),
                    "text": payload.get("text", "")
                }
                sources.append(source_obj)
            
            logger.debug(
                f"Search returned {len(sources)} results (filters={filters})"
            )
            
            return sources
            
        except UnexpectedResponse as e:
            # Collection might not exist yet
            logger.warning(f"Collection {self.collection_name} does not exist: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            raise VectorStoreError(f"Search operation failed: {str(e)}") from e

    async def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """
        Upsert vectors into the collection.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'vector', and 'payload'
            
        Raises:
            VectorStoreError: If upsert operation fails
        """
        try:
            await self.ensure_collection()
            
            # Upsert vectors
            self.client.upsert(
                collection_name=self.collection_name,
                points=vectors
            )
            logger.debug(f"Upserted {len(vectors)} vectors to {self.collection_name}")
        except Exception as e:
            logger.error(f"Error upserting vectors to Qdrant: {str(e)}")
            raise VectorStoreError(f"Upsert operation failed: {str(e)}") from e

    async def ensure_collection(self, vector_size: Optional[int] = None) -> None:
        """Ensure collection exists with correct vector size."""
        try:
            self.client.get_collection(self.collection_name)
            return
        except UnexpectedResponse:
            settings = get_settings()
            dimension = vector_size or settings.embedding_dimension or DEFAULT_EMBEDDING_DIMENSION
            logger.info(f"Creating collection {self.collection_name} with dimension {dimension}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection {self.collection_name} created successfully")
        except Exception as e:
            logger.error(f"Error ensuring collection {self.collection_name}: {str(e)}")
            raise VectorStoreError(f"Ensure collection failed: {str(e)}") from e

    async def upsert_chunks(
        self,
        document_id: str,
        source: str,
        payloads: List[Dict[str, Any]],
        vectors: List[List[float]]
    ) -> None:
        """Upsert chunk payloads + vectors into the collection."""
        try:
            if not vectors:
                logger.info(f"No vectors to upsert for document {document_id}")
                return

            await self.ensure_collection(vector_size=len(vectors[0]))

            points: List[PointStruct] = []
            for payload, vector in zip(payloads, vectors):
                point_id = payload.get("chunkId")
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.debug(
                f"Upserted {len(points)} chunks for document {document_id} to {self.collection_name}"
            )
        except Exception as e:
            logger.error(f"Error upserting chunks to Qdrant: {str(e)}")
            raise VectorStoreError(f"Upsert chunks failed: {str(e)}") from e

    async def delete_document(self, document_id: str) -> int:
        """Delete all points for a documentId, return deleted count."""
        try:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="documentId",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=query_filter,
                exact=True
            )
            deleted_count = int(getattr(count_result, "count", 0) or 0)

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=query_filter
            )
            logger.debug(
                f"Deleted {deleted_count} points for document {document_id} from {self.collection_name}"
            )
            return deleted_count
        except UnexpectedResponse as e:
            logger.warning(f"Collection {self.collection_name} does not exist: {str(e)}")
            return 0
        except Exception as e:
            logger.error(f"Error deleting document from Qdrant: {str(e)}")
            raise VectorStoreError(f"Delete document failed: {str(e)}") from e
