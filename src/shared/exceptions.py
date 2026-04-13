"""Custom exceptions for the AI Service."""


class AIServiceException(Exception):
    """Base exception for AI Service."""
    pass


class ConfigurationError(AIServiceException):
    """Raised when there's a configuration error."""
    pass


class LLMProviderError(AIServiceException):
    """Raised when LLM provider encounters an error."""
    pass


class LLMQuotaExceededError(LLMProviderError):
    """Raised when LLM API quota is exceeded."""
    pass


class VectorStoreError(AIServiceException):
    """Raised when vector store operations fail."""
    pass


class EmbeddingServiceError(AIServiceException):
    """Raised when embedding generation fails."""
    pass


class ValidationError(AIServiceException):
    """Raised when input validation fails."""
    pass


class ServiceUnavailableError(AIServiceException):
    """Raised when an external service is unavailable."""
    pass


class NotFoundError(AIServiceException):
    """Raised when a requested resource is not found."""
    pass
