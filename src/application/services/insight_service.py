"""Insight service for generating trading insights."""
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from src.domain.interfaces.llm_provider import LLMProvider
from src.application.services.prompt_builder import PromptBuilder
from src.shared.utils import normalize_insight_type
from src.shared.logging import get_logger

logger = get_logger(__name__)


class InsightService:
    """Service for generating AI-based trading insights."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize insight service.
        
        Args:
            llm_provider: LLM provider for generating insights
        """
        self.llm_provider = llm_provider
        logger.info("Initialized InsightService")

    async def generate_insight(
        self,
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        mode: str = "strict"
    ) -> Dict[str, Any]:
        """
        Generate AI-based trading insight (Buy/Sell/Hold signal) for a stock.

        Args:
            symbol: Stock symbol (e.g., VIC, VNM)
            technical_data: Technical indicators (MA, RSI, MACD, etc.)
            fundamental_data: Financial metrics (ROE, ROA, EPS, etc.)
            sentiment_data: News sentiment analysis

        Returns:
            Insight with type (Buy/Sell/Hold), confidence, reasoning, and targets
        """
        logger.info(f"Generating insight for {symbol}")
        
        # Generate insight using LLM provider
        try:
            if mode == "legacy":
                legacy_prompt = PromptBuilder.build_insight_prompt(
                    symbol=symbol,
                    technical_data=technical_data,
                    fundamental_data=fundamental_data,
                    sentiment_data=sentiment_data
                )
                response = await self.llm_provider.generate(legacy_prompt)
            else:
                system_prompt = PromptBuilder.build_insight_system_prompt()
                user_prompt = PromptBuilder.build_insight_user_prompt(
                    symbol=symbol,
                    technical_data=technical_data,
                    fundamental_data=fundamental_data,
                    sentiment_data=sentiment_data
                )
                response = await self.llm_provider.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            logger.debug(f"Received insight response for {symbol}")
        except Exception as e:
            logger.error(f"Error generating insight for {symbol}: {str(e)}")
            raise

        # Parse and structure the response
        insight = self._parse_insight_response(response, symbol, technical_data, sentiment_data)
        insight.setdefault("metadata", {})
        insight["metadata"]["rollout_mode"] = mode
        logger.info(f"Successfully generated insight for {symbol}: {insight.get('type')} with {insight.get('confidence')}% confidence")

        return insight

    def _extract_json_candidate(self, response: str) -> Optional[str]:
        """Extract JSON object from mixed model output."""
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            pass

        fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if fenced:
            return fenced.group(1)

        obj_match = re.search(r'\{.*\}', response, re.DOTALL)
        if obj_match:
            return obj_match.group(0)
        return None

    def _build_safe_fallback(self, symbol: str, flags: List[str]) -> Dict[str, Any]:
        """Build a safe deterministic fallback response."""
        return {
            "symbol": symbol,
            "type": "Hold",
            "title": "Không thể xác nhận tín hiệu mạnh",
            "description": "Dữ liệu hoặc phản hồi AI chưa đủ tin cậy để đưa khuyến nghị Buy/Sell.",
            "confidence": 40,
            "reasoning": ["Giữ trạng thái trung lập để tránh tín hiệu sai khi dữ liệu thiếu/không nhất quán."],
            "target_price": None,
            "stop_loss": None,
            "evidence": ["INSUFFICIENT_VALID_OUTPUT"],
            "metadata": {
                "prompt_version": "insight_v2",
                "model_hint": "llm_provider_default",
                "validation_notes": "|".join(flags) if flags else "safe_fallback",
                "market": "VN",
                "generated_timezone": "Asia/Ho_Chi_Minh"
            }
        }

    def _guardrail_adjustments(
        self,
        payload: Dict[str, Any],
        technical_data: Optional[Dict[str, Any]],
        sentiment_data: Optional[Dict[str, Any]],
        flags: List[str]
    ) -> Tuple[int, List[str]]:
        """Apply consistency checks and confidence adjustments."""
        confidence = payload.get("confidence", 50)
        if not isinstance(confidence, (int, float)):
            confidence = 50
            flags.append("confidence_not_numeric")
        confidence = int(max(0, min(100, confidence)))

        technical_data = technical_data or {}
        sentiment_data = sentiment_data or {}

        if len(technical_data) < 2:
            confidence = max(35, confidence - 15)
            flags.append("thin_technical_data")
        if not sentiment_data:
            confidence = max(35, confidence - 10)
            flags.append("missing_sentiment_data")

        insight_type = payload.get("type", "Hold")
        target_price = payload.get("target_price")
        stop_loss = payload.get("stop_loss")
        current_price = technical_data.get("price") or technical_data.get("current_price")
        try:
            current_price = float(current_price) if current_price is not None else None
            target_price = float(target_price) if target_price is not None else None
            stop_loss = float(stop_loss) if stop_loss is not None else None
        except (TypeError, ValueError):
            flags.append("price_field_parse_error")
            current_price = None

        if insight_type == "Buy":
            if target_price is None:
                flags.append("buy_missing_target")
                confidence = max(40, confidence - 15)
            if current_price is not None and target_price is not None and target_price < current_price:
                flags.append("buy_target_below_current")
                confidence = max(30, confidence - 25)
            if current_price is not None and stop_loss is not None and stop_loss > current_price:
                flags.append("buy_stoploss_above_current")
                confidence = max(30, confidence - 20)
        elif insight_type == "Sell":
            if target_price is not None:
                flags.append("sell_has_target_price")
                confidence = max(35, confidence - 10)
            if current_price is not None and stop_loss is not None and stop_loss < current_price:
                flags.append("sell_stoploss_below_current")
                confidence = max(30, confidence - 20)

        return confidence, flags

    def _parse_insight_response(
        self,
        response: str,
        symbol: str,
        technical_data: Optional[Dict[str, Any]],
        sentiment_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse AI response and extract insight data.
        """
        logger.debug(f"Parsing insight response for {symbol}")
        flags: List[str] = []

        try:
            json_str = self._extract_json_candidate(response)
            if not json_str:
                flags.append("json_not_found")
                return self._build_safe_fallback(symbol, flags)
            insight_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from insight response for {symbol}: {str(e)}")
            flags.append("json_decode_error")
            return self._build_safe_fallback(symbol, flags)

        # Normalize type to match enum values
        insight_type = insight_data.get("type", "Hold")
        insight_type = normalize_insight_type(insight_type)

        # Ensure reasoning is a list
        reasoning = insight_data.get("reasoning", [])
        if not isinstance(reasoning, list):
            reasoning = [str(reasoning)] if reasoning else ["Không có lý do cụ thể"]
            flags.append("reasoning_not_list")
        reasoning = [str(r).strip() for r in reasoning if str(r).strip()][:5]
        if len(reasoning) < 3:
            reasoning.append("Mức độ chắc chắn bị giới hạn do thiếu bằng chứng định lượng.")
            flags.append("reasoning_too_short")

        evidence = insight_data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)] if evidence else []
            flags.append("evidence_not_list")
        evidence = [str(e).strip() for e in evidence if str(e).strip()][:5]
        if len(evidence) < 3:
            if technical_data:
                for key in ("rsi", "macd", "volume", "trend", "price", "current_price"):
                    if technical_data.get(key) is not None:
                        evidence.append(f"{key.upper()}={technical_data.get(key)}")
            if sentiment_data and sentiment_data.get("score") is not None:
                evidence.append(f"SENTIMENT_SCORE={sentiment_data.get('score')}")
            evidence = evidence[:5]
            flags.append("evidence_autofilled")

        metadata = insight_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            flags.append("metadata_not_object")

        confidence, flags = self._guardrail_adjustments(insight_data, technical_data, sentiment_data, flags)

        # Hard downgrade to Hold for severely inconsistent output.
        severe_flags = {"buy_target_below_current", "buy_stoploss_above_current", "sell_stoploss_below_current"}
        if any(flag in severe_flags for flag in flags):
            insight_type = "Hold"
            confidence = min(confidence, 45)
            flags.append("downgraded_to_hold")

        metadata.setdefault("prompt_version", "insight_v2")
        metadata.setdefault("model_hint", "llm_provider_default")
        metadata.setdefault("market", "VN")
        metadata.setdefault("generated_timezone", "Asia/Ho_Chi_Minh")
        metadata["validation_notes"] = "|".join(flags) if flags else "ok"

        return {
            "symbol": symbol,
            "type": insight_type,
            "title": insight_data.get("title", "Khuyến nghị giao dịch"),
            "description": insight_data.get("description", "Phân tích dựa trên dữ liệu hiện có"),
            "confidence": confidence,
            "reasoning": reasoning,
            "target_price": insight_data.get("target_price"),
            "stop_loss": insight_data.get("stop_loss"),
            "evidence": evidence,
            "metadata": metadata
        }
