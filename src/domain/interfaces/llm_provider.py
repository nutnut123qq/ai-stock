from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Generate content using the LLM.
        
        Args:
            prompt: The input prompt (used if system_prompt and user_prompt are not provided)
            system_prompt: Optional system prompt for instruction
            user_prompt: Optional user prompt (used with system_prompt)
            
        Returns:
            Generated text content
        """
        pass

