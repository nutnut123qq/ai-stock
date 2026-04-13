"""Embedding service for generating text embeddings."""
from typing import Optional
from sentence_transformers import SentenceTransformer
import asyncio
from functools import lru_cache
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.shared.config import get_settings
from src.shared.exceptions import EmbeddingServiceError
from src.shared.constants import DEFAULT_EMBEDDING_MODEL
from src.shared.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService(EmbeddingProvider):
    """Service for generating text embeddings using sentence transformers."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                       If None, uses default from configuration.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name or DEFAULT_EMBEDDING_MODEL
        self._model: Optional[SentenceTransformer] = None
        logger.info(f"Initialized EmbeddingService with model: {self.model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Embedding model {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {str(e)}")
                raise EmbeddingServiceError(
                    f"Failed to load embedding model: {str(e)}"
                ) from e
        return self._model

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            EmbeddingServiceError: If embedding generation fails
        """
        try:
            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.model.encode,
                text
            )
            embedding_list = embedding.tolist()
            logger.debug(f"Generated embedding of dimension {len(embedding_list)}")
            return embedding_list
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise EmbeddingServiceError(
                f"Failed to generate embedding: {str(e)}"
            ) from e

    def clear_cache(self) -> None:
        """Clear the model cache (useful for testing or memory management)."""
        self._model = None
        logger.debug("Embedding model cache cleared")
