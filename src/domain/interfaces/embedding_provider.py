"""Embedding provider interface for text embeddings."""
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for generating text embeddings."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            Embedding vector as list of floats
        """
        pass
