"""Stock Data API routes."""
from typing import Annotated, List, Optional

from fastapi import APIRouter, Body, Depends, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
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


class MultipleQuotesRequest(BaseModel):
    """Request model for multiple stock quotes."""
    symbols: List[str] = Field(..., min_items=1, max_items=50, description="Danh sách mã chứng khoán (tối đa 50)")


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
    symbols_data = await run_in_threadpool(stock_service.get_all_symbols, exchange)
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
    source: str = Query("KBS", description="Nguồn dữ liệu: KBS, VCI"),
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
    quote = await run_in_threadpool(stock_service.get_stock_quote, symbol, source)
    return quote


@router.post("/quotes", response_model=List[StockQuoteResponse])
async def get_multiple_quotes(
    request: MultipleQuotesRequest,
    source: str = Query("KBS", description="Nguồn dữ liệu: KBS, VCI"),
    stock_service: StockDataService = Depends(get_stock_data_service)
):
    """
    Lấy giá của nhiều mã chứng khoán cùng lúc.

    Args:
        request: Request containing list of stock symbols (max 50)
        source: Data source
        stock_service: Stock data service instance

    Returns:
        List of stock quotes
    """
    quotes = await run_in_threadpool(stock_service.get_multiple_quotes, request.symbols, source)
    return quotes


@router.get("/history/{symbol}", response_model=List[HistoricalDataPoint])
async def get_historical_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)"),
    interval: str = Query("1D", description="Khoảng thời gian: 1D, 1W, 1M"),
    source: str = Query("KBS", description="Nguồn dữ liệu: KBS, VCI"),
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
    
    data = await run_in_threadpool(
        stock_service.get_historical_data,
        symbol,
        start_date,
        end_date,
        interval,
        source,
    )
    return data


@router.get("/vn30")
async def get_vn30_quotes(
    source: str = Query("KBS", description="Nguồn dữ liệu: KBS, VCI"),
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
        'ACB', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB',
        'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB',
        'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VPL', 'VRE'
    ]
    
    quotes = await run_in_threadpool(stock_service.get_multiple_quotes, vn30_symbols, source)
    return quotes
