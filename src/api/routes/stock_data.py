"""Stock Data API routes."""
from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from src.application.services.stock_data_service import StockDataService
from src.api.dependencies import get_stock_data_service

router = APIRouter(prefix="/api/stock", tags=["stock"])


class StockQuoteResponse(BaseModel):
    """Response model for stock quote."""
    symbol: str
    currentPrice: float
    previousClose: float
    change: float
    changePercent: float
    volume: int
    high: float
    low: float
    open: float
    lastUpdated: str
    priceUnit: str = "VND"


class SymbolInfo(BaseModel):
    """Response model for symbol information."""
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    industry: Optional[str] = None


class HistoricalDataPoint(BaseModel):
    """Response model for historical data point."""
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    priceUnit: str = "VND"


class SymbolsResponse(BaseModel):
    """Response model for symbols list."""
    symbols: List[SymbolInfo]


@router.get("/symbols", response_model=SymbolsResponse)
async def get_all_symbols(
    exchange: Optional[str] = Query(None, description="Sàn giao dịch: HOSE, HNX, UPCOM"),
    stock_service: StockDataService = Depends(get_stock_data_service)
):
    """
    Lấy danh sách tất cả mã chứng khoán.

    Args:
        exchange: Exchange filter (optional)
        stock_service: Stock data service instance

    Returns:
        List of stock symbols
    """
    symbols_data = stock_service.get_all_symbols(exchange)
    symbols = [
        SymbolInfo(
            symbol=s.get('ticker', s.get('symbol', '')),
            name=s.get('organ_name', s.get('company_name', s.get('name'))),
            exchange=s.get('exchange', ''),
            industry=s.get('icb_name3', s.get('industry'))
        )
        for s in symbols_data
    ]
    return SymbolsResponse(symbols=symbols)


@router.get("/quote/{symbol}", response_model=StockQuoteResponse)
async def get_stock_quote(
    symbol: str,
    source: str = Query("VCI", description="Nguồn dữ liệu: VCI, TCBS"),
    stock_service: StockDataService = Depends(get_stock_data_service)
):
    """
    Lấy giá hiện tại của một mã chứng khoán.

    Args:
        symbol: Stock symbol
        source: Data source
        stock_service: Stock data service instance

    Returns:
        Stock quote information
    """
    quote = stock_service.get_stock_quote(symbol, source)
    return quote


@router.post("/quotes", response_model=List[StockQuoteResponse])
async def get_multiple_quotes(
    symbols: List[str],
    source: str = Query("VCI", description="Nguồn dữ liệu: VCI, TCBS"),
    stock_service: StockDataService = Depends(get_stock_data_service)
):
    """
    Lấy giá của nhiều mã chứng khoán cùng lúc.

    Args:
        symbols: List of stock symbols
        source: Data source
        stock_service: Stock data service instance

    Returns:
        List of stock quotes
    """
    quotes = stock_service.get_multiple_quotes(symbols, source)
    return quotes


@router.get("/history/{symbol}", response_model=List[HistoricalDataPoint])
async def get_historical_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)"),
    interval: str = Query("1D", description="Khoảng thời gian: 1D, 1W, 1M"),
    source: str = Query("VCI", description="Nguồn dữ liệu: VCI, TCBS"),
    stock_service: StockDataService = Depends(get_stock_data_service)
):
    """
    Lấy dữ liệu lịch sử của mã chứng khoán.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        interval: Time interval
        source: Data source
        stock_service: Stock data service instance

    Returns:
        List of historical data points
    """
    # Nếu không có start_date, lấy 30 ngày gần nhất
    if not start_date:
        start = datetime.now() - timedelta(days=30)
        start_date = start.strftime('%Y-%m-%d')
    
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = stock_service.get_historical_data(symbol, start_date, end_date, interval, source)
    return data


@router.get("/vn30")
async def get_vn30_quotes(
    source: str = Query("VCI", description="Nguồn dữ liệu: VCI, TCBS"),
    stock_service: StockDataService = Depends(get_stock_data_service)
):
    """
    Lấy giá của các mã VN30.

    Args:
        source: Data source
        stock_service: Stock data service instance

    Returns:
        List of VN30 stock quotes
    """
    # Danh sách mã VN30 (có thể cập nhật định kỳ)
    vn30_symbols = [
        'VIC', 'VNM', 'VCB', 'VRE', 'VHM', 'GAS', 'MSN', 'BID', 'CTG', 'HPG',
        'TCB', 'MBB', 'VPB', 'PLX', 'SAB', 'VJC', 'GVR', 'FPT', 'POW', 'SSI',
        'MWG', 'HDB', 'ACB', 'TPB', 'STB', 'PDR', 'VIB', 'BCM', 'KDH', 'NVL'
    ]
    
    quotes = stock_service.get_multiple_quotes(vn30_symbols, source)
    return quotes
