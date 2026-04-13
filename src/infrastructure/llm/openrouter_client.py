"""OpenRouter (OpenAI-compatible) client for LLMProvider."""
from typing import Optional

from openai import OpenAI

from src.domain.interfaces.llm_provider import LLMProvider
from src.shared.config import get_settings
from src.shared.exceptions import LLMProviderError
from src.shared.logging import get_logger

logger = get_logger(__name__)


class OpenRouterClient(LLMProvider):
    """OpenRouter API — e.g. openrouter/free Free Models Router."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        api_key = settings.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        base_url = (settings.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self.model_name = model_name or settings.openrouter_model or "openrouter/free"
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        default_headers: dict[str, str] = {}
        if settings.openrouter_http_referer:
            default_headers["HTTP-Referer"] = settings.openrouter_http_referer
        if settings.openrouter_app_title:
            default_headers["X-Title"] = settings.openrouter_app_title

        kwargs: dict = {
            "api_key": api_key,
            "base_url": base_url,
        }
        if default_headers:
            kwargs["default_headers"] = default_headers

        self.client = OpenAI(**kwargs)
        logger.info("Initialized OpenRouterClient with model: %s", self.model_name)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt and user_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMProviderError("OpenRouter returned empty content")
            return content
        except Exception as e:
            logger.error("OpenRouter error: %s", e)
            raise LLMProviderError(f"Error generating content with OpenRouter: {e!s}") from e
