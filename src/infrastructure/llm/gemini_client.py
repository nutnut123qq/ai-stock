"""Google Gemini client implementation for LLM provider."""
from typing import Optional
from google import genai
from src.domain.interfaces.llm_provider import LLMProvider
from src.shared.config import get_settings
from src.shared.exceptions import LLMProviderError
from src.shared.logging import get_logger

logger = get_logger(__name__)


class GeminiClient(LLMProvider):
    """Google Gemini client implementing LLMProvider interface."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            model_name: Specific model to use. Defaults to 'gemini-2.5-flash'.
        """
        settings = get_settings()
        api_key = settings.gemini_api_key
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        
        # Use specified model or default
        if model_name:
            self.model_name = model_name
        else:
            # Use gemini-2.5-flash as default
            self.model_name = "gemini-2.5-flash"
        
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        logger.info(f"Initialized GeminiClient with model: {self.model_name}")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, user_prompt: Optional[str] = None) -> str:
        """
        Generate content using Gemini model.
        
        Args:
            prompt: The input prompt (used if system_prompt and user_prompt are not provided)
            system_prompt: Optional system prompt for instruction
            user_prompt: Optional user prompt (used with system_prompt)
            
        Returns:
            Generated text content
            
        Raises:
            LLMProviderError: For LLM provider errors
        """
        try:
            # If system_prompt and user_prompt are provided, combine them
            if system_prompt and user_prompt:
                # Gemini supports system instructions via system_instruction parameter
                # For now, we'll combine them in the prompt
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            elif system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            logger.debug(f"Generating with Gemini model: {self.model_name}")
            
            # Generate content using Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
            
            # Extract text from response
            # The response structure may vary, try different ways to access text
            content = None
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content'):
                    if hasattr(candidate.content, 'parts') and len(candidate.content.parts) > 0:
                        content = candidate.content.parts[0].text
                    elif hasattr(candidate.content, 'text'):
                        content = candidate.content.text
            elif hasattr(response, 'content'):
                if hasattr(response.content, 'parts') and len(response.content.parts) > 0:
                    content = response.content.parts[0].text
                elif hasattr(response.content, 'text'):
                    content = response.content.text
            
            if content is None:
                raise LLMProviderError("Unexpected response format from Gemini API - could not extract text")
            
            logger.debug(f"Successfully generated response with Gemini model: {self.model_name}")
            return content
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error generating content with Gemini {self.model_name}: {error_str}")
            raise LLMProviderError(f"Error generating content with Gemini: {str(e)}") from e
