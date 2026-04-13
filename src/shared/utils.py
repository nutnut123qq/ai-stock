"""Utility functions for the AI Service."""
import re
from typing import List, Dict, Any, Optional
from src.shared.constants import (
    TREND_UP_KEYWORDS, TREND_DOWN_KEYWORDS,
    CONFIDENCE_HIGH_KEYWORDS, CONFIDENCE_LOW_KEYWORDS,
    CONFIDENCE_HIGH_SCORE, CONFIDENCE_MEDIUM_SCORE, CONFIDENCE_LOW_SCORE,
    RECOMMENDATION_BUY_KEYWORDS, RECOMMENDATION_SELL_KEYWORDS,
    SENTIMENT_POSITIVE_KEYWORDS, SENTIMENT_NEGATIVE_KEYWORDS,
    INSIGHT_TYPE_MAPPING
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


def extract_list_items(text: str, keywords: List[str]) -> List[str]:
    """
    Extract list items from text based on keywords.
    
    Args:
        text: Text to extract from
        keywords: Keywords to search for section
        
    Returns:
        List of extracted items
    """
    items = []
    lines = text.split('\n')
    in_section = False
    
    for line in lines:
        # Check if we're in the relevant section
        if any(keyword in line.lower() for keyword in keywords):
            in_section = True
            continue
        
        # Extract bullet points or numbered items
        if in_section:
            # Stop at next section
            if line.strip().startswith('#') or (
                line.strip() and 
                line.strip()[0].isdigit() and 
                '.' in line[:3]
            ):
                if not any(keyword in line.lower() for keyword in keywords):
                    in_section = False
                    continue
            
            # Extract item
            match = re.match(r'^[\s\-\*\d\.]+(.+)$', line)
            if match:
                item = match.group(1).strip()
                if item and len(item) > 5:
                    items.append(item)
    
    return items


def extract_trend(response: str) -> str:
    """
    Extract trend from AI response.
    
    Args:
        response: AI response text
        
    Returns:
        Trend: "Up", "Down", or "Sideways"
    """
    response_lower = response.lower()
    
    if any(word in response_lower for word in TREND_UP_KEYWORDS):
        return "Up"
    elif any(word in response_lower for word in TREND_DOWN_KEYWORDS):
        return "Down"
    else:
        return "Sideways"


def extract_confidence(response: str) -> tuple[str, float]:
    """
    Extract confidence level from AI response.
    
    Args:
        response: AI response text
        
    Returns:
        Tuple of (confidence_level, confidence_score)
    """
    response_lower = response.lower()
    
    if any(word in response_lower for word in CONFIDENCE_HIGH_KEYWORDS):
        return "High", CONFIDENCE_HIGH_SCORE
    elif any(word in response_lower for word in CONFIDENCE_LOW_KEYWORDS):
        return "Low", CONFIDENCE_LOW_SCORE
    else:
        return "Medium", CONFIDENCE_MEDIUM_SCORE


def extract_recommendation(response: str) -> str:
    """
    Extract recommendation from AI response.
    
    Args:
        response: AI response text
        
    Returns:
        Recommendation: "Buy", "Hold", or "Sell"
    """
    response_lower = response.lower()
    
    if any(word in response_lower for word in RECOMMENDATION_BUY_KEYWORDS):
        return "Buy"
    elif any(word in response_lower for word in RECOMMENDATION_SELL_KEYWORDS):
        return "Sell"
    else:
        return "Hold"


def extract_sentiment(text: str) -> str:
    """
    Extract sentiment from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment: "positive", "negative", or "neutral"
    """
    text_lower = text.lower()
    
    positive_count = sum(
        1 for keyword in SENTIMENT_POSITIVE_KEYWORDS
        if keyword in text_lower
    )
    negative_count = sum(
        1 for keyword in SENTIMENT_NEGATIVE_KEYWORDS
        if keyword in text_lower
    )
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


def normalize_insight_type(insight_type: str) -> str:
    """
    Normalize insight type to standard values.
    
    Args:
        insight_type: Raw insight type
        
    Returns:
        Normalized type: "Buy", "Sell", or "Hold"
    """
    return INSIGHT_TYPE_MAPPING.get(insight_type.lower(), "Hold")


def format_technical_data(technical_data: Optional[Dict[str, Any]]) -> str:
    """
    Format technical data for prompt.
    
    Args:
        technical_data: Technical indicators dict
        
    Returns:
        Formatted string for prompt
    """
    if not technical_data:
        return ""
    
    return f"""1. CHỈ SỐ KỸ THUẬT:
- MA (Moving Average): {technical_data.get('ma', 'N/A')}
- RSI (Relative Strength Index): {technical_data.get('rsi', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}
- Bollinger Bands: {technical_data.get('bollinger', 'N/A')}
- Volume: {technical_data.get('volume', 'N/A')}
- Price Trend: {technical_data.get('trend', 'N/A')}

"""


def format_fundamental_data(fundamental_data: Optional[Dict[str, Any]]) -> str:
    """
    Format fundamental data for prompt.
    
    Args:
        fundamental_data: Fundamental metrics dict
        
    Returns:
        Formatted string for prompt
    """
    if not fundamental_data:
        return ""
    
    return f"""2. CHỈ SỐ TÀI CHÍNH:
- ROE (Return on Equity): {fundamental_data.get('roe', 'N/A')}%
- ROA (Return on Assets): {fundamental_data.get('roa', 'N/A')}%
- EPS (Earnings Per Share): {fundamental_data.get('eps', 'N/A')}
- P/E Ratio: {fundamental_data.get('pe', 'N/A')}
- Revenue Growth: {fundamental_data.get('revenue_growth', 'N/A')}%
- Profit Margin: {fundamental_data.get('profit_margin', 'N/A')}%

"""


def format_sentiment_data(sentiment_data: Optional[Dict[str, Any]]) -> str:
    """
    Format sentiment data for prompt.
    
    Args:
        sentiment_data: Sentiment analysis dict
        
    Returns:
        Formatted string for prompt
    """
    if not sentiment_data:
        return ""
    
    return f"""3. PHÂN TÍCH TÂM LÝ THỊ TRƯỜNG:
- Sentiment Score: {sentiment_data.get('score', 'N/A')}
- News Sentiment: {sentiment_data.get('sentiment', 'N/A')}
- Social Media Buzz: {sentiment_data.get('social_buzz', 'N/A')}
- Recent News: {sentiment_data.get('recent_news', 'N/A')}

"""
