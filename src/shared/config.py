"""Centralized configuration management for the AI Service."""
import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    blackbox_api_key: Optional[str] = Field(default=None, env="BLACKBOX_API_KEY")
    blackbox_model: Optional[str] = Field(default=None, env="BLACKBOX_MODEL")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        env="OPENROUTER_BASE_URL",
    )
    openrouter_model: str = Field(default="openrouter/free", env="OPENROUTER_MODEL")
    openrouter_http_referer: Optional[str] = Field(default=None, env="OPENROUTER_HTTP_REFERER")
    openrouter_app_title: Optional[str] = Field(default=None, env="OPENROUTER_APP_TITLE")
    openrouter_rpm_limit: int = Field(default=18, env="OPENROUTER_RPM_LIMIT")
    openrouter_base_delay: float = Field(default=3.5, env="OPENROUTER_BASE_DELAY")
    beeknoee_api_key: Optional[str] = Field(default=None, env="BEEKNOEE_API_KEY")
    beeknoee_base_url: str = Field(
        default="https://platform.beeknoee.com/api/v1",
        env="BEEKNOEE_BASE_URL",
    )
    beeknoee_model: str = Field(default="glm-4.7-flash", env="BEEKNOEE_MODEL")
    api_title: str = Field(default="Stock Investment AI Service", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    
    # Vector Store Configuration
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_collection_name: str = Field(
        default="stock_documents", 
        env="QDRANT_COLLECTION_NAME"
    )
    
    # Message Queue Configuration (optional)
    rabbitmq_connection_string: Optional[str] = Field(
        default=None,
        env="RABBITMQ_CONNECTION_STRING"
    )
    
    # LLM Configuration
    default_llm_model: str = Field(
        default="blackboxai/openai/gpt-4-turbo",
        env="DEFAULT_LLM_MODEL"
    )
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")
    insight_shadow_mode: bool = Field(default=False, env="INSIGHT_SHADOW_MODE")
    insight_canary_ratio: float = Field(default=1.0, env="INSIGHT_CANARY_RATIO")
    insight_prompt_version: str = Field(default="insight_v2", env="INSIGHT_PROMPT_VERSION")
    
    # Embedding Configuration
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL_NAME"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="json",
        env="LOG_FORMAT"
    )  # json or text
    
    # Internal API Key for RAG endpoints
    internal_api_key: Optional[str] = Field(
        default=None,
        env="INTERNAL_API_KEY"
    )

    # Redis cache configuration (shared infra with .NET backend — use a different DB index to avoid key collisions)
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=1, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_socket_timeout: float = Field(default=2.0, env="REDIS_SOCKET_TIMEOUT")

    # VNStock caching knobs (fail-open when Redis unreachable)
    vnstock_cache_enabled: bool = Field(default=True, env="VNSTOCK_CACHE_ENABLED")
    vnstock_cache_quote_ttl: int = Field(default=45, env="VNSTOCK_CACHE_QUOTE_TTL")
    vnstock_cache_history_ttl: int = Field(default=21600, env="VNSTOCK_CACHE_HISTORY_TTL")
    vnstock_cache_symbols_ttl: int = Field(default=86400, env="VNSTOCK_CACHE_SYMBOLS_TTL")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("insight_canary_ratio")
    @classmethod
    def validate_canary_ratio(cls, v):
        """Validate canary ratio is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Insight canary ratio must be between 0 and 1")
        return v
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env that are not in this model


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
