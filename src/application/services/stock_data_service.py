"""Stock Data Service - Lấy dữ liệu chứng khoán từ vnstock."""
from typing import List, Optional, Dict, Any, Callable, TypeVar
from datetime import datetime, timedelta
import pandas as pd
import random
import time
try:
    from vnstock_data import Listing as SponsorListing, Quote as SponsorQuote
    HAS_VNSTOCK_SPONSOR = True
except ImportError:
    SponsorListing = None
    SponsorQuote = None
    HAS_VNSTOCK_SPONSOR = False
from vnstock import Vnstock, Listing as CommunityListing
from src.shared.logging import get_logger
from src.shared.exceptions import ServiceUnavailableError, NotFoundError
from src.infrastructure.cache.redis_cache import RedisCacheService

logger = get_logger(__name__)
T = TypeVar("T")
DEFAULT_SOURCE = "KBS"
VCI_SOURCE = "VCI"

# Default TTLs tuned so that the .NET `StockPriceUpdateJob` (60s cron) always
# misses at least once per run, while history / symbol lookups are held longer
# because they rarely change intraday.
DEFAULT_QUOTE_TTL_SECONDS = 45
DEFAULT_HISTORY_TTL_SECONDS = 6 * 3600
DEFAULT_SYMBOLS_TTL_SECONDS = 24 * 3600

_CACHE_KEY_PREFIX = "ai:vnstock"


class _SponsorStockClient:
    """Provide the same `.quote.history(...)` shape used by the service."""

    def __init__(self, symbol: str, source: str):
        self.quote = SponsorQuote(symbol=symbol, source=source)


class StockDataService:
    """Service để lấy dữ liệu chứng khoán từ vnstock."""

    def __init__(
        self,
        cache: Optional[RedisCacheService] = None,
        cache_enabled: bool = True,
        quote_ttl: int = DEFAULT_QUOTE_TTL_SECONDS,
        history_ttl: int = DEFAULT_HISTORY_TTL_SECONDS,
        symbols_ttl: int = DEFAULT_SYMBOLS_TTL_SECONDS,
    ) -> None:
        self.listing = SponsorListing(source=DEFAULT_SOURCE) if HAS_VNSTOCK_SPONSOR else CommunityListing()
        self._client = Vnstock()
        self._cache: Optional[RedisCacheService] = cache
        self._cache_enabled: bool = bool(cache_enabled and cache is not None)
        self._quote_ttl = int(quote_ttl)
        self._history_ttl = int(history_ttl)
        self._symbols_ttl = int(symbols_ttl)
        self._max_retries = 2
        self._base_backoff_seconds = 0.6

    # ----- Cache helpers -------------------------------------------------

    @staticmethod
    def _cache_key(kind: str, *parts: Any) -> str:
        """Build a namespaced cache key in upper-case for consistency.

        Example: ``_cache_key("quote", "vic", "kbs")`` -> ``"ai:vnstock:quote:VIC:KBS"``.
        """
        normalized = ":".join(str(p).upper() for p in parts if p is not None and str(p) != "")
        if normalized:
            return f"{_CACHE_KEY_PREFIX}:{kind}:{normalized}"
        return f"{_CACHE_KEY_PREFIX}:{kind}"

    def _cache_get(self, key: str) -> Optional[Any]:
        if not self._cache_enabled or self._cache is None:
            return None
        return self._cache.get(key)

    def _cache_set(self, key: str, value: Any, ttl: int) -> None:
        if not self._cache_enabled or self._cache is None:
            return
        self._cache.set(key, value, ttl)

    def _build_stock_client(self, symbol: str, source: str):
        if HAS_VNSTOCK_SPONSOR:
            return _SponsorStockClient(symbol=symbol.upper(), source=source)
        return self._client.stock(symbol=symbol.upper(), source=source)

    def _execute_with_retry(self, action: Callable[[], T], operation_name: str) -> T:
        """Retry short transient transport errors from vnstock with jitter."""
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return action()
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                delay = self._base_backoff_seconds * (2 ** attempt) + random.uniform(0, 0.25)
                logger.warning(
                    "Retrying %s after %.2fs (attempt %d/%d): %s",
                    operation_name,
                    delay,
                    attempt + 1,
                    self._max_retries,
                    str(exc),
                )
                time.sleep(delay)
        raise last_error if last_error else RuntimeError(f"{operation_name} failed")

    @staticmethod
    def _should_fallback_to_kbs(source: str, error: Exception) -> bool:
        return source.upper() == VCI_SOURCE and "'data'" in str(error)

    def _fetch_history_with_fallback(
        self,
        symbol: str,
        source: str,
        start: str,
        end: str,
        interval: str,
        operation_name: str,
    ) -> pd.DataFrame:
        normalized_source = source.upper()
        stock = self._build_stock_client(symbol=symbol, source=normalized_source)
        try:
            return self._execute_with_retry(
                lambda: stock.quote.history(start=start, end=end, interval=interval),
                operation_name,
            )
        except Exception as exc:
            if not self._should_fallback_to_kbs(normalized_source, exc):
                raise

            logger.warning(
                "Primary source %s failed for %s with schema error (%s); retrying with %s",
                normalized_source,
                operation_name,
                str(exc),
                DEFAULT_SOURCE,
            )
            fallback_stock = self._build_stock_client(symbol=symbol, source=DEFAULT_SOURCE)
            fallback_operation = f"{operation_name}:fallback:{DEFAULT_SOURCE}"
            return self._execute_with_retry(
                lambda: fallback_stock.quote.history(start=start, end=end, interval=interval),
                fallback_operation,
            )

    @staticmethod
    def _normalize_vnd_price(raw_price: float) -> float:
        """
        Chuẩn hóa đơn vị giá về VND đầy đủ.
        vnstock thường trả đơn vị nghìn VND (ví dụ 160 => 160000 VND).
        """
        if raw_price <= 0:
            return raw_price
        return raw_price * 1000 if raw_price < 1000 else raw_price
    
    def get_all_symbols(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả mã chứng khoán.

        Args:
            exchange: Sàn giao dịch (HOSE, HNX, UPCOM)

        Returns:
            List of stock symbols with basic info.

        Raises:
            ServiceUnavailableError: If unable to fetch symbols from data source.
        """
        cache_key = self._cache_key("symbols", exchange or "ALL")
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("cache hit %s", cache_key)
            return cached

        try:
            df = self.listing.all_symbols()

            if exchange:
                df = df[df['exchange'] == exchange.upper()]

            symbols = df.to_dict('records')

            self._cache_set(cache_key, symbols, self._symbols_ttl)
            return symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            raise ServiceUnavailableError(f"Failed to fetch stock symbols: {str(e)}") from e
    
    def get_stock_quote(self, symbol: str, source: str = DEFAULT_SOURCE) -> Dict[str, Any]:
        """
        Lấy giá hiện tại của mã chứng khoán.

        Args:
            symbol: Mã chứng khoán (VD: VIC, VNM).
            source: Nguồn dữ liệu (KBS, VCI).

        Returns:
            Dict chứa thông tin giá hiện tại.

        Raises:
            NotFoundError: If symbol not found or no data available.
            ServiceUnavailableError: If unable to fetch data from source.
        """
        normalized_source = source.upper()
        cache_key = self._cache_key("quote", symbol, normalized_source)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("cache hit %s", cache_key)
            return cached

        try:
            # Lấy dữ liệu 2 ngày gần nhất để tính change.
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)

            df = self._fetch_history_with_fallback(
                symbol=symbol,
                source=normalized_source,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1D',
                operation_name=f"get_stock_quote:{symbol.upper()}:{normalized_source}",
            )

            if df.empty:
                raise NotFoundError(f"Không tìm thấy dữ liệu cho mã {symbol}")

            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest

            current_price = self._normalize_vnd_price(float(latest['close']))
            previous_close = self._normalize_vnd_price(float(previous['close']))
            quote = {
                'symbol': symbol.upper(),
                'currentPrice': current_price,
                'previousClose': previous_close,
                'change': current_price - previous_close,
                'changePercent': float((float(latest['close']) - float(previous['close'])) / float(previous['close']) * 100),
                'volume': int(latest['volume']) if 'volume' in latest else 0,
                'high': self._normalize_vnd_price(float(latest['high'])),
                'low': self._normalize_vnd_price(float(latest['low'])),
                'open': self._normalize_vnd_price(float(latest['open'])),
                'lastUpdated': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name),
                'priceUnit': 'VND'
            }

            # Only positive responses are cached — propagate NotFound/unavailable so
            # the upstream can retry without a stale negative.
            self._cache_set(cache_key, quote, self._quote_ttl)
            return quote
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            raise ServiceUnavailableError(f"Failed to fetch stock quote for {symbol}: {str(e)}") from e
    
    def get_multiple_quotes(self, symbols: List[str], source: str = DEFAULT_SOURCE) -> List[Dict[str, Any]]:
        """
        Lấy giá của nhiều mã chứng khoán
        
        Args:
            symbols: Danh sách mã chứng khoán
            source: Nguồn dữ liệu
        
        Returns:
            List of stock quotes
        """
        quotes = []
        for symbol in symbols:
            quote = self.get_stock_quote(symbol, source)
            if quote:
                quotes.append(quote)
            # A tiny spacing helps avoid burst traffic to upstream.
            time.sleep(0.05)
        
        return quotes
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1D',
        source: str = DEFAULT_SOURCE
    ) -> List[Dict[str, Any]]:
        """
        Lấy dữ liệu lịch sử
        
        Args:
            symbol: Mã chứng khoán
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            interval: Khoảng thời gian (1D, 1W, 1M)
            source: Nguồn dữ liệu
        
        Returns:
            List of historical data points
            
        Raises:
            NotFoundError: If symbol not found or no data available
            ServiceUnavailableError: If unable to fetch data from source
        """
        normalized_source = source.upper()
        cache_key = self._cache_key(
            "history", symbol, normalized_source, start_date, end_date, interval
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("cache hit %s", cache_key)
            return cached

        try:
            df = self._fetch_history_with_fallback(
                symbol=symbol,
                source=normalized_source,
                start=start_date,
                end=end_date,
                interval=interval,
                operation_name=f"get_historical_data:{symbol.upper()}:{normalized_source}",
            )

            if df.empty:
                raise NotFoundError(f"Không tìm thấy dữ liệu lịch sử cho mã {symbol}")

            data = df.reset_index().to_dict('records')

            for item in data:
                if 'time' in item and hasattr(item['time'], 'isoformat'):
                    item['time'] = item['time'].isoformat()
                for field in ('open', 'high', 'low', 'close'):
                    if field in item and item[field] is not None:
                        item[field] = self._normalize_vnd_price(float(item[field]))
                item['priceUnit'] = 'VND'

            self._cache_set(cache_key, data, self._history_ttl)
            return data
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            raise ServiceUnavailableError(f"Failed to fetch historical data for {symbol}: {str(e)}") from e

