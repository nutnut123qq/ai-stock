"""Prompt builder utilities for AI services."""
import json
from typing import Optional, Dict, Any
from src.shared.constants import TIME_HORIZON_MAP
from src.shared.utils import (
    format_technical_data,
    format_fundamental_data,
    format_sentiment_data
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Utility class for building AI prompts."""
    
    @staticmethod
    def build_forecast_prompt(
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        time_horizon: str = "short"
    ) -> str:
        """
        Build prompt for stock forecast.
        
        Args:
            symbol: Stock symbol
            technical_data: Technical indicators
            fundamental_data: Fundamental metrics
            sentiment_data: Sentiment analysis
            time_horizon: Forecast time period (short, medium, long)
            
        Returns:
            Formatted prompt string
        """
        time_period = TIME_HORIZON_MAP.get(time_horizon, "1-5 ngày tới")
        
        prompt = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam. Hãy dự báo xu hướng cổ phiếu {symbol} trong {time_period}.

DỮ LIỆU PHÂN TÍCH:

"""
        
        # Add technical analysis
        prompt += format_technical_data(technical_data)
        
        # Add fundamental analysis
        prompt += format_fundamental_data(fundamental_data)
        
        # Add sentiment analysis
        prompt += format_sentiment_data(sentiment_data)
        
        prompt += """HÃY CUNG CẤP DỰ BÁO CHI TIẾT:

1. **Xu hướng dự báo**: Tăng/Giảm/Đi ngang
2. **Mức độ tin cậy**: Cao (>70%) / Trung bình (50-70%) / Thấp (<50%)
3. **Mục tiêu giá**:
   - Giá mục tiêu (target price)
   - Giá hỗ trợ (support level)
   - Giá kháng cự (resistance level)
4. **Yếu tố chính** (2-3 yếu tố quan trọng nhất)
5. **Rủi ro** (2-3 rủi ro cần lưu ý)
6. **Khuyến nghị**: Mua/Giữ/Bán

Trả lời bằng tiếng Việt, có cấu trúc rõ ràng và dễ hiểu."""
        
        return prompt
    
    @staticmethod
    def build_insight_prompt(
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for trading insight.
        
        Args:
            symbol: Stock symbol
            technical_data: Technical indicators
            fundamental_data: Fundamental metrics
            sentiment_data: Sentiment analysis
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam. Hãy đưa ra khuyến nghị giao dịch (MUA/BÁN/GIỮ) cho cổ phiếu {symbol} dựa trên dữ liệu phân tích.

DỮ LIỆU PHÂN TÍCH:

"""
        
        # Add technical analysis
        if technical_data:
            prompt += f"""1. CHỈ SỐ KỸ THUẬT:
- MA (Moving Average): {technical_data.get('ma', 'N/A')}
- RSI (Relative Strength Index): {technical_data.get('rsi', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}
- Volume: {technical_data.get('volume', 'N/A')}
- Price Trend: {technical_data.get('trend', 'N/A')}

"""
        
        # Add fundamental analysis
        if fundamental_data:
            prompt += f"""2. CHỈ SỐ TÀI CHÍNH:
- ROE (Return on Equity): {fundamental_data.get('roe', 'N/A')}%
- ROA (Return on Assets): {fundamental_data.get('roa', 'N/A')}%
- EPS (Earnings Per Share): {fundamental_data.get('eps', 'N/A')}
- P/E Ratio: {fundamental_data.get('pe', 'N/A')}
- Revenue Growth: {fundamental_data.get('revenue_growth', 'N/A')}%

"""
        
        # Add sentiment analysis
        if sentiment_data:
            prompt += f"""3. TÂM LÝ THỊ TRƯỜNG:
- Sentiment Score: {sentiment_data.get('score', 'N/A')}
- Overall Sentiment: {sentiment_data.get('sentiment', 'N/A')}
- Recent News: {sentiment_data.get('recent_news', 'N/A')}

"""
        
        prompt += """YÊU CẦU:
Hãy phân tích và đưa ra khuyến nghị giao dịch với format JSON sau:

{
  "type": "Buy" hoặc "Sell" hoặc "Hold",
  "title": "Tiêu đề ngắn gọn của insight (ví dụ: 'Strong Buy Signal Detected')",
  "description": "Mô tả ngắn gọn về tín hiệu (1-2 câu)",
  "confidence": 0-100 (điểm tin cậy),
  "reasoning": ["Lý do 1", "Lý do 2", "Lý do 3"] (danh sách các yếu tố chính),
  "target_price": giá mục tiêu nếu là Buy (optional, có thể null),
  "stop_loss": giá cắt lỗ nếu là Buy hoặc Sell (optional, có thể null)
}

Lưu ý:
- "Buy": Cổ phiếu có tiềm năng tăng giá mạnh
- "Sell": Cổ phiếu có nguy cơ giảm giá hoặc nên chốt lời
- "Hold": Cổ phiếu ổn định, không có tín hiệu rõ ràng
- Confidence: 0-100, càng cao càng tin cậy
- Reasoning: Liệt kê 3-5 yếu tố chính ảnh hưởng đến quyết định
"""
        
        return prompt

    @staticmethod
    def build_insight_system_prompt() -> str:
        """Build strict system prompt for insight generation."""
        return """Bạn là chuyên gia phân tích chứng khoán Việt Nam.
Nhiệm vụ: đưa ra khuyến nghị Buy/Sell/Hold với độ chính xác và tính nhất quán cao.

Quy tắc bắt buộc:
1) Chỉ trả về JSON hợp lệ, không thêm markdown/code fence/text ngoài JSON.
2) JSON phải có đầy đủ keys:
   type,title,description,confidence,reasoning,target_price,stop_loss,evidence,metadata
3) type chỉ được là Buy | Sell | Hold.
4) confidence là số nguyên 0-100.
5) reasoning phải có 3-5 ý, ngắn gọn, bám dữ liệu.
6) evidence phải có ít nhất 3 bằng chứng có số liệu cụ thể (RSI/MACD/volume/sentiment...).
7) Tính nhất quán:
   - Nếu tín hiệu mâu thuẫn, giảm confidence.
   - Buy thường có target_price >= giá hiện tại và stop_loss <= giá hiện tại.
   - Sell thường có stop_loss >= giá hiện tại nếu là mức vô hiệu hóa luận điểm giảm.
8) metadata phải chứa:
   prompt_version, model_hint, validation_notes, market, generated_timezone
9) Nếu dữ liệu thiếu nhiều, ưu tiên Hold và nêu rõ hạn chế trong reasoning/metadata.
"""

    @staticmethod
    def build_insight_user_prompt(
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build user prompt with structured data and few-shot examples."""
        technical_data = technical_data or {}
        fundamental_data = fundamental_data or {}
        sentiment_data = sentiment_data or {}

        return f"""Phân tích cổ phiếu {symbol} theo dữ liệu sau:

Technical:
{json.dumps(technical_data, ensure_ascii=False)}

Fundamental:
{json.dumps(fundamental_data, ensure_ascii=False)}

Sentiment:
{json.dumps(sentiment_data, ensure_ascii=False)}

Few-shot ví dụ chuẩn (tham khảo format và logic nhất quán):

Ví dụ 1 (Buy):
{{
  "type": "Buy",
  "title": "Động lượng tăng được củng cố",
  "description": "Tín hiệu kỹ thuật và dòng tiền ủng hộ kịch bản tăng ngắn hạn.",
  "confidence": 74,
  "reasoning": [
    "RSI ở vùng trung tính-tích cực, chưa quá mua",
    "MACD duy trì phân kỳ dương",
    "Khối lượng cao hơn trung bình 20 phiên"
  ],
  "target_price": 102500,
  "stop_loss": 94500,
  "evidence": [
    "RSI=58.4",
    "MACD_hist=+0.7",
    "VolumeRatio20D=1.32"
  ],
  "metadata": {{
    "prompt_version": "insight_v2",
    "model_hint": "deterministic_json",
    "validation_notes": "consistent_buy_setup",
    "market": "VN",
    "generated_timezone": "Asia/Ho_Chi_Minh"
  }}
}}

Ví dụ 2 (Sell):
{{
  "type": "Sell",
  "title": "Rủi ro giảm giá ngắn hạn tăng cao",
  "description": "Xung lực suy yếu và tâm lý tin tức tiêu cực làm tăng xác suất điều chỉnh.",
  "confidence": 71,
  "reasoning": [
    "RSI suy giảm và tiến gần vùng yếu",
    "MACD cắt xuống, histogram âm mở rộng",
    "Tin tức gần đây thiên tiêu cực"
  ],
  "target_price": null,
  "stop_loss": 98700,
  "evidence": [
    "RSI=41.2",
    "MACD_hist=-0.9",
    "SentimentScore=-0.34"
  ],
  "metadata": {{
    "prompt_version": "insight_v2",
    "model_hint": "deterministic_json",
    "validation_notes": "bearish_alignment",
    "market": "VN",
    "generated_timezone": "Asia/Ho_Chi_Minh"
  }}
}}

Ví dụ 3 (Hold):
{{
  "type": "Hold",
  "title": "Tín hiệu phân hóa, chưa đủ lợi thế vị thế mới",
  "description": "Dữ liệu kỹ thuật và tâm lý chưa đồng thuận để nâng xác suất quyết định Buy/Sell.",
  "confidence": 56,
  "reasoning": [
    "RSI trung tính, thiếu động lượng rõ ràng",
    "MACD sát ngưỡng 0, tín hiệu nhiễu",
    "Sentiment gần trung lập"
  ],
  "target_price": null,
  "stop_loss": null,
  "evidence": [
    "RSI=50.1",
    "MACD_hist=0.02",
    "SentimentScore=0.03"
  ],
  "metadata": {{
    "prompt_version": "insight_v2",
    "model_hint": "deterministic_json",
    "validation_notes": "mixed_signal_hold",
    "market": "VN",
    "generated_timezone": "Asia/Ho_Chi_Minh"
  }}
}}

Trả về đúng một JSON object theo schema bắt buộc.
"""
