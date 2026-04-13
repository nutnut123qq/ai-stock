"""Constants used throughout the AI Service."""

# LLM Model Configuration
AVAILABLE_BLACKBOX_MODELS = [
    'blackboxai/openai/gpt-5.2',
    'blackboxai/openai/gpt-4',
    'blackboxai/openai/gpt-4-turbo',
    'blackboxai/openai/gpt-3.5-turbo',
]

# Default Model Settings
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_MAX_TOKENS = 2048

# Vector Store Configuration
DEFAULT_QDRANT_COLLECTION_NAME = "stock_documents"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# API Configuration
DEFAULT_API_TITLE = "Stock Investment AI Service"
DEFAULT_API_VERSION = "1.0.0"

# Time Horizon Mapping
TIME_HORIZON_MAP = {
    "short": "1-5 ngày tới",
    "medium": "1-4 tuần tới",
    "long": "1-3 tháng tới"
}

# Confidence Levels
CONFIDENCE_HIGH_THRESHOLD = 70
CONFIDENCE_MEDIUM_THRESHOLD = 50
CONFIDENCE_HIGH_SCORE = 75.0
CONFIDENCE_MEDIUM_SCORE = 50.0
CONFIDENCE_LOW_SCORE = 35.0

# Trend Keywords
TREND_UP_KEYWORDS = ["tăng", "up", "bullish", "tích cực"]
TREND_DOWN_KEYWORDS = ["giảm", "down", "bearish", "tiêu cực"]

# Confidence Keywords
CONFIDENCE_HIGH_KEYWORDS = ["cao", "high", ">70", "mạnh"]
CONFIDENCE_LOW_KEYWORDS = ["thấp", "low", "<50", "yếu"]

# Recommendation Keywords
RECOMMENDATION_BUY_KEYWORDS = ["mua", "buy", "tích lũy"]
RECOMMENDATION_SELL_KEYWORDS = ["bán", "sell", "thoát"]

# Sentiment Keywords
SENTIMENT_POSITIVE_KEYWORDS = ["positive", "tích cực", "tăng", "tốt", "khả quan", "lạc quan"]
SENTIMENT_NEGATIVE_KEYWORDS = ["negative", "tiêu cực", "giảm", "xấu", "bi quan", "lo ngại"]

# Insight Type Mapping
INSIGHT_TYPE_MAPPING = {
    "buy": "Buy",
    "sell": "Sell",
    "hold": "Hold",
    "mua": "Buy",
    "bán": "Sell",
    "giữ": "Hold"
}

# Quota Error Patterns
QUOTA_ERROR_PATTERNS = ['429', 'quota', 'Quota exceeded', 'rate limit']

# HTTP Status Codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
