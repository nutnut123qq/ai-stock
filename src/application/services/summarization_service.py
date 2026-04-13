"""Summarization service for news articles."""
import json
from typing import Dict, Any
from src.domain.interfaces.llm_provider import LLMProvider
from src.shared.utils import extract_sentiment
from src.shared.logging import get_logger

logger = get_logger(__name__)


class SummarizationService:
    """Service for summarizing news articles."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize summarization service.
        
        Args:
            llm_provider: LLM provider for generating summaries
        """
        self.llm_provider = llm_provider
        logger.info("Initialized SummarizationService")

    async def summarize(self, content: str) -> Dict[str, Any]:
        """
        Summarize news content and extract sentiment.
        
        Args:
            content: News article content
            
        Returns:
            Dictionary with summary, sentiment, impact_assessment, and key_points
        """
        logger.info(f"Summarizing content: {len(content)} characters")
        
        prompt = f"""Bạn là chuyên gia phân tích tài chính. Hãy phân tích bài viết tin tức sau về thị trường chứng khoán Việt Nam:

{content}

Hãy trả lời theo định dạng JSON sau:
{{
    "summary": "Tóm tắt ngắn gọn 2-3 câu về nội dung chính",
    "sentiment": "positive/negative/neutral",
    "impact_assessment": "Đánh giá tác động đến thị trường/cổ phiếu liên quan",
    "key_points": ["Điểm chính 1", "Điểm chính 2", "Điểm chính 3"]
}}

Lưu ý:
- Sentiment: positive (tích cực), negative (tiêu cực), neutral (trung lập)
- Impact assessment: Phân tích cụ thể tác động đến giá cổ phiếu, xu hướng thị trường
- Key points: Các điểm quan trọng nhất trong bài viết
"""

        try:
            response = await self.llm_provider.generate(prompt)
            logger.debug("Received summarization response")

            result = self._parse_summary_response(response)
            logger.info(f"Successfully summarized content with sentiment: {result.get('sentiment')}")

            return result
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            raise

    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """
        Parse summary response from LLM.
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed summary dictionary
        """
        try:
            # Try to parse JSON response
            # Remove markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            result = json.loads(json_str)

            # Validate required fields
            if "summary" not in result:
                result["summary"] = "Không thể tạo tóm tắt"
            if "sentiment" not in result:
                result["sentiment"] = "neutral"
            if "impact_assessment" not in result:
                result["impact_assessment"] = "Chưa có đánh giá tác động"
            if "key_points" not in result:
                result["key_points"] = []

            # Normalize sentiment value
            sentiment_lower = result["sentiment"].lower()
            if "positive" in sentiment_lower or "tích cực" in sentiment_lower:
                result["sentiment"] = "positive"
            elif "negative" in sentiment_lower or "tiêu cực" in sentiment_lower:
                result["sentiment"] = "negative"
            else:
                result["sentiment"] = "neutral"

            return result

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from summary response, using fallback parsing")
            # Fallback to simple parsing if JSON parsing fails
            summary = self._extract_summary(response)
            sentiment = extract_sentiment(response)
            impact = self._extract_impact(response)

            return {
                "summary": summary,
                "sentiment": sentiment,
                "impact_assessment": impact,
                "key_points": []
            }

    def _extract_summary(self, text: str) -> str:
        """Extract summary from unstructured text."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Return first non-empty line as summary
            return lines[0][:500]  # Limit to 500 chars
        return "Không thể tạo tóm tắt"

    def _extract_impact(self, text: str) -> str:
        """Extract impact assessment from unstructured text."""
        # Look for impact-related sentences
        sentences = text.split('.')
        for sentence in sentences:
            if any(
                keyword in sentence.lower() 
                for keyword in ["impact", "tác động", "ảnh hưởng", "affect"]
            ):
                return sentence.strip()

        return "Chưa có đánh giá tác động cụ thể"
