"""
Service for answering questions with provided context parts.
V1 Minimal: NO RAG, NO Qdrant, just LLM with structured context.
"""
import re
import logging
from typing import List, Optional, Dict, Any
from src.domain.interfaces.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class AnswerContextService:
    """
    Service for Q&A with provided context parts (for Analysis Reports).
    
    P0 Fixes:
    - #13: Strict regex r'\\[(\\d{1,2})\\]' to avoid false positives like [2024]
    - #14: Fallback to [0] if LLM forgets to cite sources
    """
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
    
    async def answer_question(
        self,
        question: str,
        context_parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Answer a question based on provided context parts.
        
        Args:
            question: User's question
            context_parts: List of context parts with source_type, title, excerpt, etc.
        
        Returns:
            Dict with 'answer' (str) and 'used_sources' (list of int indices)
        """
        if not context_parts:
            raise ValueError("Context parts cannot be empty")
        
        # 1. Build numbered context string for LLM
        context_text = self._build_numbered_context(context_parts)
        
        # 2. Build prompt with strict citation instruction (P0 Fix #13)
        system_prompt = """You are a financial analysis assistant.
Answer questions based ONLY on the provided context.
When referencing information, cite sources using EXACTLY this format: [1], [2], [3], etc.
- Use square brackets with single number only
- Separate multiple citations: [1] [2] (not [1,2] or [1-2])
- ALWAYS cite at least one source for factual claims
- Place citations at the end of sentences
- Do NOT use brackets for years like [2024] or [999]"""
        
        user_prompt = f"""Context:
{context_text}

Question: {question}

Please answer the question and cite sources using [1], [2] notation."""
        
        logger.info(f"Answering question with {len(context_parts)} context parts")
        
        # 3. Call LLM
        try:
            answer = await self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
        
        # 4. Extract used source indices with strict validation (P0 Fix #13, #14)
        used_sources = self._extract_citations(answer, len(context_parts))
        
        logger.info(f"Generated answer with {len(used_sources)} citations")
        
        return {
            "answer": answer,
            "used_sources": used_sources
        }
    
    def _build_numbered_context(self, context_parts: List[Dict[str, Any]]) -> str:
        """
        Build numbered context string for LLM.
        
        Format:
        [1] Title
        Source: analysis_report
        URL: https://...
        Content: excerpt...
        --------------------------------------------------
        [2] Title
        ...
        """
        context_text = ""
        for idx, part in enumerate(context_parts):
            context_text += f"\n[{idx + 1}] {part.get('title', 'Untitled')}\n"
            context_text += f"Source: {part.get('source_type', 'unknown')}\n"
            
            if part.get('url'):
                context_text += f"URL: {part['url']}\n"
            
            context_text += f"Content: {part.get('excerpt', '')}\n"
            context_text += "-" * 50 + "\n"
        
        return context_text
    
    def _extract_citations(self, answer: str, context_parts_count: int) -> List[int]:
        """
        Extract citation indices from answer, with strict validation.
        Only accept [1], [2], ..., [N] where N = len(context_parts).
        Fallback to [0] if no valid citations found.
        
        P0 Fix #13: Only match 1-2 digit numbers to avoid false positives like [2024]
        P0 Fix #14: Fallback to first source if LLM forgets to cite
        
        Args:
            answer: LLM answer text
            context_parts_count: Number of context parts provided
        
        Returns:
            List of 0-based indices (sorted, unique)
        """
        # ✅ P0 Fix #13: Only match 1-2 digit numbers in brackets
        raw_matches = re.findall(r'\[(\d{1,2})\]', answer)
        
        # Convert to integers
        nums = []
        for x in raw_matches:
            if x.isdigit():
                nums.append(int(x))
        
        # ✅ P0 Fix #13: Range check - only accept 1..context_parts_count
        valid_indices = set()
        for n in nums:
            if 1 <= n <= context_parts_count:
                valid_indices.add(n - 1)  # Convert to 0-based index
        
        # ✅ P0 Fix #14: Fallback to first source (report) if none found
        if not valid_indices:
            logger.warning(f"No valid citations found in answer. Falling back to [0]")
            return [0]
        
        # Sort and return
        return sorted(list(valid_indices))
