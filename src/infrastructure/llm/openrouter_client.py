"""OpenRouter (OpenAI-compatible) client for LLMProvider."""
import asyncio
import time
from typing import Optional

from openai import OpenAI

from src.domain.interfaces.llm_provider import LLMProvider
from src.infrastructure.llm.rate_limiter import SlidingWindowRateLimiter
from src.shared.config import get_settings
from src.shared.exceptions import LLMProviderError
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Retry configuration
OPENROUTER_MAX_RETRIES = 3
OPENROUTER_BASE_DELAY = 1.5  # seconds


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

        # Sliding-window rate limiter for OpenRouter free tier (default 18 req/min)
        rpm = getattr(settings, "openrouter_rpm_limit", 18)
        self._rate_limiter = SlidingWindowRateLimiter(max_requests=rpm, window_seconds=60.0)
        logger.info(
            "Initialized OpenRouterClient with model: %s (rpm_limit=%d)",
            self.model_name,
            rpm,
        )

    def _invoke_once(self, messages: list) -> str:
        """Single blocking call to OpenRouter."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise LLMProviderError("OpenRouter returned empty content")
        return content

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

        last_error: Optional[Exception] = None
        for attempt in range(1, OPENROUTER_MAX_RETRIES + 1):
            try:
                await self._rate_limiter.acquire(label=f"OpenRouter attempt {attempt}/{OPENROUTER_MAX_RETRIES}")
                content = await asyncio.to_thread(self._invoke_once, messages)
                self._rate_limiter.record()
                if attempt > 1:
                    logger.info("OpenRouter succeeded on attempt %d/%d", attempt, OPENROUTER_MAX_RETRIES)
                return content
            except LLMProviderError as e:
                last_error = e
                logger.warning("OpenRouter empty content on attempt %d/%d: %s", attempt, OPENROUTER_MAX_RETRIES, e)
            except Exception as e:
                last_error = e
                logger.warning("OpenRouter error on attempt %d/%d: %s", attempt, OPENROUTER_MAX_RETRIES, e)

            if attempt < OPENROUTER_MAX_RETRIES:
                delay = OPENROUTER_BASE_DELAY * (2 ** (attempt - 1))
                logger.info("Retrying OpenRouter in %.1fs...", delay)
                await asyncio.sleep(delay)

        logger.error("OpenRouter failed after %d attempts: %s", OPENROUTER_MAX_RETRIES, last_error)
        raise LLMProviderError(
            f"Error generating content with OpenRouter after {OPENROUTER_MAX_RETRIES} attempts: {last_error!s}"
        ) from last_error
