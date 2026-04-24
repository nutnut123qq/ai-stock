"""Financial data supplement from Yahoo Finance (yfinance)."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
from src.shared.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class YFinanceFinancialResponse(BaseModel):
    symbol: str
    period: str
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_profit: Optional[float] = None
    net_profit: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    equity: Optional[float] = None
    total_assets: Optional[float] = None


@router.get("/financial/yfinance/{symbol}", response_model=YFinanceFinancialResponse)
async def get_yfinance_financial(symbol: str):
    """
    Fetch latest quarterly financial data from Yahoo Finance for a VN stock.
    Symbol should be the raw ticker (e.g., VIC, HPG) — .VN suffix is appended internally.
    """
    ticker_symbol = f"{symbol.upper()}.VN"
    try:
        stock = yf.Ticker(ticker_symbol)
        income = stock.quarterly_financials
        balance = stock.quarterly_balance_sheet

        if income is None or income.empty:
            logger.warning(f"No quarterly financial data for {ticker_symbol}")
            return YFinanceFinancialResponse(symbol=symbol, period="N/A")

        latest_col = income.columns[0]
        period = str(latest_col.date())

        def get_val(df, row_name):
            if df is not None and row_name in df.index:
                val = df.loc[row_name, latest_col]
                return float(val) if val is not None else None
            return None

        revenue = get_val(income, "Total Revenue")
        gross_profit = get_val(income, "Gross Profit")
        operating_profit = get_val(income, "Operating Income")
        net_profit = get_val(income, "Net Income")
        eps = get_val(income, "Basic EPS")
        equity = get_val(balance, "Stockholders Equity")
        total_assets = get_val(balance, "Total Assets")

        # Calculate ROE / ROA when both numerator and denominator exist
        roe = None
        roa = None
        if net_profit and equity and equity != 0:
            roe = round((net_profit / equity) * 100, 4)
        if net_profit and total_assets and total_assets != 0:
            roa = round((net_profit / total_assets) * 100, 4)

        return YFinanceFinancialResponse(
            symbol=symbol,
            period=period,
            revenue=revenue,
            gross_profit=gross_profit,
            operating_profit=operating_profit,
            net_profit=net_profit,
            eps=eps,
            roe=roe,
            roa=roa,
            equity=equity,
            total_assets=total_assets,
        )
    except Exception as e:
        logger.error(f"Error fetching yfinance data for {ticker_symbol}: {e}")
        return YFinanceFinancialResponse(symbol=symbol, period="error")
