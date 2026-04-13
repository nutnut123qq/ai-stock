"""Blackbox AI client implementation for LLM provider."""
import random
from typing import Optional
from openai import OpenAI
from src.domain.interfaces.llm_provider import LLMProvider
from src.shared.config import get_settings
from src.shared.exceptions import LLMQuotaExceededError, LLMProviderError
from src.shared.constants import AVAILABLE_BLACKBOX_MODELS, QUOTA_ERROR_PATTERNS
from src.shared.logging import get_logger

logger = get_logger(__name__)


class BlackboxClient(LLMProvider):
    """Blackbox AI client implementing LLMProvider interface."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Blackbox client.
        
        Args:
            model_name: Specific model to use. If None, randomly selects from available models.
        """
        settings = get_settings()
        api_key = settings.blackbox_api_key
        
        if not api_key:
            raise ValueError("BLACKBOX_API_KEY environment variable is not set")
        
        # Initialize OpenAI client with Blackbox API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.blackbox.ai"
        )
        
        # Use specified model or pick a random one from available models
        if model_name:
            if model_name not in AVAILABLE_BLACKBOX_MODELS:
                logger.warning(f"Model {model_name} not in available models, using default")
                model_name = AVAILABLE_BLACKBOX_MODELS[0]
            self.model_name = model_name
        else:
            self.model_name = random.choice(AVAILABLE_BLACKBOX_MODELS)
        
        self.current_model_index = AVAILABLE_BLACKBOX_MODELS.index(
            self.model_name
        ) if self.model_name in AVAILABLE_BLACKBOX_MODELS else 0
        
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        logger.info(f"Initialized BlackboxClient with model: {self.model_name}")

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Generate content using current model, with automatic fallback to other models if quota exceeded.
        
        Args:
            prompt: The input prompt (used if system_prompt and user_prompt are not provided)
            system_prompt: Optional system prompt for instruction
            user_prompt: Optional user prompt (used with system_prompt)
            
        Returns:
            Generated text content
            
        Raises:
            LLMQuotaExceededError: When all models have quota exceeded
            LLMProviderError: For other LLM provider errors
        """
        # Build messages based on provided prompts
        messages = []
        if system_prompt and user_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})
        
        last_error = None
        models_tried = [self.model_name]
        
        # Try current model first
        try:
            logger.debug(f"Generating with model: {self.model_name}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            logger.debug(f"Successfully generated response with model: {self.model_name}")
            return content
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.warning(f"Error with model {self.model_name}: {error_str}")
            
            # Check if it's a quota error
            is_quota_error = any(
                pattern.lower() in error_str.lower() 
                for pattern in QUOTA_ERROR_PATTERNS
            )
            
            if not is_quota_error:
                # Not a quota error, raise immediately
                logger.error(f"Non-quota error with {self.model_name}: {error_str}")
                raise LLMProviderError(f"Error generating content with {self.model_name}: {str(e)}") from e
        
        # If quota exceeded, try other models
        available_indices = [
            i for i in range(len(AVAILABLE_BLACKBOX_MODELS))
            if AVAILABLE_BLACKBOX_MODELS[i] not in models_tried
        ]
        
        if not available_indices:
            # All models tried, raise original error
            logger.error("All models exhausted. No available models to try.")
            raise LLMQuotaExceededError(
                f"All models quota exceeded. Last error: {str(last_error)}"
            ) from last_error
        
        # Try remaining models in order
        for idx in available_indices:
            try:
                model_name = AVAILABLE_BLACKBOX_MODELS[idx]
                logger.info(f"Trying fallback model: {model_name}")
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                # Update current model if successful
                self.model_name = model_name
                self.current_model_index = idx
                content = response.choices[0].message.content
                logger.info(f"Successfully generated with fallback model: {model_name}")
                return content
            except Exception as e:
                error_str = str(e)
                is_quota_error = any(
                    pattern.lower() in error_str.lower()
                    for pattern in QUOTA_ERROR_PATTERNS
                )
                
                if not is_quota_error:
                    # Not a quota error, raise immediately
                    logger.error(f"Non-quota error with {model_name}: {error_str}")
                    raise LLMProviderError(
                        f"Error generating content with {model_name}: {str(e)}"
                    ) from e
                
                models_tried.append(model_name)
                last_error = e
                logger.warning(f"Quota exceeded for model {model_name}, trying next")
        
        # All models exhausted due to quota
        logger.error("All models quota exceeded. No available models.")
        raise LLMQuotaExceededError(
            f"All models quota exceeded. Last error: {str(last_error)}"
        ) from last_error
    
    def rotate_model(self) -> None:
        """Rotate to next available model."""
        self.current_model_index = (self.current_model_index + 1) % len(AVAILABLE_BLACKBOX_MODELS)
        self.model_name = AVAILABLE_BLACKBOX_MODELS[self.current_model_index]
        logger.info(f"Rotated to model: {self.model_name}")
