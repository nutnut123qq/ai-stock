"""Beeknoee Platform (OpenAI-compatible) client for LLMProvider."""

import asyncio
import os
import re
import time
from typing import Optional

from openai import OpenAI, RateLimitError

from src.domain.interfaces.llm_provider import LLMProvider
from src.infrastructure.llm.beeknoee_sync import BEEKNOEE_API_LOCK, cooldown
from src.shared.config import get_settings
from src.shared.exceptions import LLMProviderError
from src.shared.logging import get_logger

logger = get_logger(__name__)


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    """Extract a Retry-After delay (in seconds) from a rate-limit error.

    Looks at the HTTP Retry-After header first, then falls back to parsing
    the provider message (e.g. ``Retry after 8s``).
    """
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers is not None:
                raw = headers.get("retry-after") or headers.get("Retry-After")
                if raw:
                    return float(str(raw).strip())
    except (ValueError, TypeError):
        pass

    match = re.search(r"retry\s*after\s*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


class BeeknoeeClient(LLMProvider):
    """Beeknoee unified API — same wire format as OpenAI chat completions."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        api_key = settings.beeknoee_api_key
        if not api_key:
            raise ValueError("BEEKNOEE_API_KEY environment variable is not set")

        base_url = (
            settings.beeknoee_base_url or "https://platform.beeknoee.com/api/v1"
        ).rstrip("/")
        self.model_name = model_name or settings.beeknoee_model or "glm-4.7-flash"
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        # Beeknoee free tier rejects overlapping calls (1/1 concurrent); OpenAI SDK
        # default retries would start a second request while the first is still active.
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        logger.info("Initialized BeeknoeeClient with model: %s", self.model_name)

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

        respect_retry_after = os.getenv("BEEKNOEE_RESPECT_RETRY_AFTER", "true").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        try:
            max_retry_after = float(os.getenv("BEEKNOEE_MAX_RETRY_AFTER_SEC", "30") or 30)
        except ValueError:
            max_retry_after = 30.0

        def _invoke_once() -> str:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMProviderError("Beeknoee returned empty content")
            return content

        def _sync_call() -> str:
            # Hold BEEKNOEE_API_LOCK across the entire round-trip so LangGraph
            # node invocations and REST routes share a single in-flight slot.
            with BEEKNOEE_API_LOCK:
                try:
                    try:
                        return _invoke_once()
                    except RateLimitError as rle:
                        if not respect_retry_after:
                            raise

                        wait_seconds = _extract_retry_after_seconds(rle)
                        if wait_seconds is None or wait_seconds <= 0:
                            raise
                        wait_seconds = min(wait_seconds, max_retry_after)

                        logger.warning(
                            "Beeknoee rate-limited, waited %.1fs before retry",
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                        return _invoke_once()
                finally:
                    cooldown()

        try:
            return await asyncio.to_thread(_sync_call)
        except LLMProviderError:
            raise
        except Exception as e:
            logger.error("Beeknoee error: %s", e)
            raise LLMProviderError(
                f"Error generating content with Beeknoee: {e!s}"
            ) from e
