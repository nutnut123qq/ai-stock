"""Custom exceptions for the AI Service."""


class AIServiceException(Exception):
    """Base exception for AI Service.
    
    All custom exceptions in the AI service inherit from this class.
    Catch this to handle any AI-service-specific error generically.
    """
    pass


class ConfigurationError(AIServiceException):
    """Raised when a required configuration value is missing or invalid.
    
    Typically indicates missing environment variables or misconfigured
    settings at startup. The application should fail fast and log the
    exact missing key server-side.
    """
    pass


class LLMProviderError(AIServiceException):
    """Raised when an LLM provider operation fails.
    
    This includes network errors, malformed responses, or provider-side
    failures. Callers should treat this as a transient failure and may
    retry with exponential backoff or fall back to another provider.
    """
    pass


class LLMQuotaExceededError(LLMProviderError):
    """Raised when the LLM provider quota or rate limit is exceeded.
    
    Indicates the API key has hit a token limit, request cap, or
    throughput restriction. Callers should back off exponentially
    before retrying and consider alerting operators.
    """
    pass


class VectorStoreError(AIServiceException):
    """Raised when a vector store operation fails.
    
    Covers connection errors, query failures, write timeouts, or
    collection misconfiguration in Qdrant. Treat as transient unless
    the connection is completely lost across multiple retries.
    """
    pass


class EmbeddingServiceError(AIServiceException):
    """Raised when text embedding generation fails.
    
    Usually caused by model loading issues, GPU OOM, or input size
    exceeding the model's limit. Callers may truncate input and retry.
    """
    pass


class ValidationError(AIServiceException):
    """Raised when user input fails validation.
    
    Use this for request payload errors (e.g. missing fields, invalid
    formats, out-of-range values). The client should receive a 422
    response with a clear, safe error message.
    """
    pass


class ServiceUnavailableError(AIServiceException):
    """Raised when an external dependency is temporarily unavailable.
    
    This is a catch-all for downstream services (Redis, RQ, backend API)
    that are unreachable. Callers should return a 503 to the client.
    """
    pass


class NotFoundError(AIServiceException):
    """Raised when a requested resource does not exist.
    
    Use for missing documents, unknown symbols, or expired jobs.
    Maps naturally to an HTTP 404 response.
    """
    pass
