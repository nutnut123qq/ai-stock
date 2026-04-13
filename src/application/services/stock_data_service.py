"""Stock Data Service - Lấy dữ liệu chứng khoán từ vnstock."""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from vnstock import Vnstock, Listing
from src.shared.logging import get_logger
from src.shared.exceptions import ServiceUnavailableError, NotFoundError

logger = get_logger(__name__)


class StockDataService:
    """Service để lấy dữ liệu chứng khoán từ vnstock"""
    
    def __init__(self):
        self.listing = Listing()
        self._cache = {}
        self._cache_ttl = 60  # Cache 60 giây cho real-time data

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
        Lấy danh sách tất cả mã chứng khoán
        
        Args:
            exchange: Sàn giao dịch (HOSE, HNX, UPCOM)
        
        Returns:
            List of stock symbols with basic info
            
        Raises:
            ServiceUnavailableError: If unable to fetch symbols from data source
        """
        try:
            df = self.listing.all_symbols()
            
            if exchange:
                df = df[df['exchange'] == exchange.upper()]
            
            # Convert DataFrame to list of dicts
            symbols = df.to_dict('records')
            
            return symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            raise ServiceUnavailableError(f"Failed to fetch stock symbols: {str(e)}") from e
    
    def get_stock_quote(self, symbol: str, source: str = 'VCI') -> Dict[str, Any]:
        """
        Lấy giá hiện tại của mã chứng khoán
        
        Args:
            symbol: Mã chứng khoán (VD: VIC, VNM)
            source: Nguồn dữ liệu (VCI, TCBS)
        
        Returns:
            Dict chứa thông tin giá hiện tại
            
        Raises:
            NotFoundError: If symbol not found or no data available
            ServiceUnavailableError: If unable to fetch data from source
        """
        try:
            stock = Vnstock().stock(symbol=symbol.upper(), source=source)
            
            # Lấy dữ liệu 2 ngày gần nhất để tính change
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            df = stock.quote.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1D'
            )
            
            if df.empty:
                raise NotFoundError(f"Không tìm thấy dữ liệu cho mã {symbol}")
            
            # Lấy dòng cuối cùng (ngày gần nhất)
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            current_price = self._normalize_vnd_price(float(latest['close']))
            previous_close = self._normalize_vnd_price(float(previous['close']))
            return {
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
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            raise ServiceUnavailableError(f"Failed to fetch stock quote for {symbol}: {str(e)}") from e
    
    def get_multiple_quotes(self, symbols: List[str], source: str = 'VCI') -> List[Dict[str, Any]]:
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
        
        return quotes
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1D',
        source: str = 'VCI'
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
        try:
            stock = Vnstock().stock(symbol=symbol.upper(), source=source)
            df = stock.quote.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise NotFoundError(f"Không tìm thấy dữ liệu lịch sử cho mã {symbol}")
            
            # Convert DataFrame to list of dicts
            data = df.reset_index().to_dict('records')
            
            # Convert datetime to string + normalize all prices to VND
            for item in data:
                if 'time' in item and hasattr(item['time'], 'isoformat'):
                    item['time'] = item['time'].isoformat()
                for field in ('open', 'high', 'low', 'close'):
                    if field in item and item[field] is not None:
                        item[field] = self._normalize_vnd_price(float(item[field]))
                item['priceUnit'] = 'VND'
            
            return data
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            raise ServiceUnavailableError(f"Failed to fetch historical data for {symbol}: {str(e)}") from e

