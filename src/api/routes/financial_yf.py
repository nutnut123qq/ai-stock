"""Financial data supplement for VN stocks.

Primary source: vnstock_data (paid API) for Vietnamese stocks.
Fallback: Yahoo Finance (yfinance) if vnstock_data is unavailable or fails.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import math

import yfinance as yf
from src.shared.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

try:
    from vnstock_data import Finance
    HAS_VNSTOCK_DATA = True
except ImportError:  # pragma: no cover
    HAS_VNSTOCK_DATA = False
    Finance = None  # type: ignore[misc,assignment]


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


def _to_float(val):
    """Convert a scalar to Python float, treating NaN/inf as None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _find_column(columns, candidates):
    """Find the first matching column name (case-insensitive)."""
    col_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in col_map:
            return col_map[key]
    return None


def _extract_vnstock_data(symbol: str) -> Optional[YFinanceFinancialResponse]:
    """Extract latest quarterly financials using vnstock_data."""
    if not HAS_VNSTOCK_DATA or Finance is None:
        return None

    fin = Finance(symbol=symbol.upper(), source="VCI")

    # Fetch income statement (mandatory)
    income = fin.income_statement(period="quarter", lang="vi")
    if income is None or income.empty:
        return None

    # Latest row is the last row (most recent quarter)
    latest = income.iloc[-1]
    period = str(income.index[-1]) if income.index is not None else "N/A"

    # --- Revenue ---
    rev_col = _find_column(
        income.columns,
        [
            "Doanh thu thuần",
            "Thu nhập lãi thuần",
            "Tổng thu nhập hoạt động",
            "Doanh thu bán hàng và cung cấp dịch vụ",
        ],
    )
    revenue = _to_float(latest.get(rev_col)) if rev_col else None

    # --- Gross Profit ---
    gp_col = _find_column(income.columns, ["Lợi nhuận gộp"])
    gross_profit = _to_float(latest.get(gp_col)) if gp_col else None

    # --- Operating Profit ---
    op_col = _find_column(
        income.columns,
        [
            "Lãi/(lỗ) từ hoạt động kinh doanh",
            "Lợi nhuận thuần hoạt động trước khi trích lập dự phòng tổn thất tín dụng",
            "Lãi/(lỗ) thuần từ hoạt động khác",
        ],
    )
    operating_profit = _to_float(latest.get(op_col)) if op_col else None

    # --- Net Profit ---
    np_col = _find_column(
        income.columns,
        [
            "Lãi/(lỗ) thuần sau thuế",
            "Lợi nhuận sau thuế",
            "Lợi nhuận của Cổ đông của Công ty mẹ",
            "Cổ đông của Công ty mẹ",
        ],
    )
    net_profit = _to_float(latest.get(np_col)) if np_col else None

    # --- EPS ---
    eps_col = _find_column(
        income.columns,
        [
            "Lãi cơ bản trên cổ phiếu (VND)",
            "Lãi cơ bản trên cổ phiếu",
        ],
    )
    eps = _to_float(latest.get(eps_col)) if eps_col else None

    # --- Balance sheet (best-effort) ---
    equity = None
    total_assets = None
    try:
        balance = fin.balance_sheet(period="quarter", lang="vi")
        if balance is not None and not balance.empty:
            bal_latest = balance.iloc[-1]
            eq_col = _find_column(
                balance.columns,
                ["Vốn chủ sở hữu", "VỐN CHỦ SỞ HỮU", "Vốn chủ sở hữu "],
            )
            equity = _to_float(bal_latest.get(eq_col)) if eq_col else None

            ta_col = _find_column(
                balance.columns,
                ["TỔNG CỘNG TÀI SẢN", "Tổng cộng tài sản", "TỔNG TÀI SẢN", "Tổng tài sản"],
            )
            total_assets = _to_float(bal_latest.get(ta_col)) if ta_col else None
    except Exception as exc:
        logger.warning("vnstock_data balance_sheet failed for %s: %s", symbol, exc)

    # --- Ratio report (best-effort) ---
    roe = None
    roa = None
    try:
        ratio = fin.ratio(period="quarter", lang="vi")
        if ratio is not None and not ratio.empty:
            ratio_latest = ratio.iloc[-1]
            roe_col = _find_column(ratio.columns, ["ROE (%)", "ROE"])
            roa_col = _find_column(ratio.columns, ["ROA (%)", "ROA"])
            roe = _to_float(ratio_latest.get(roe_col)) if roe_col else None
            roa = _to_float(ratio_latest.get(roa_col)) if roa_col else None
    except Exception as exc:
        logger.warning("vnstock_data ratio failed for %s: %s", symbol, exc)

    # Fallback calculation if ratio missing but we have net_profit + equity/assets
    if roe is None and net_profit and equity and equity != 0:
        roe = round((net_profit / equity) * 100, 4)
    if roa is None and net_profit and total_assets and total_assets != 0:
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


def _extract_yfinance_data(symbol: str) -> YFinanceFinancialResponse:
    """Original Yahoo Finance fallback."""
    ticker_symbol = f"{symbol.upper()}.VN"
    stock = yf.Ticker(ticker_symbol)
    income = stock.quarterly_financials
    balance = stock.quarterly_balance_sheet

    if income is None or income.empty:
        logger.warning("No quarterly financial data for %s", ticker_symbol)
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


@router.get("/financial/yfinance/{symbol}", response_model=YFinanceFinancialResponse)
async def get_yfinance_financial(symbol: str):
    """
    Fetch latest quarterly financial data for a VN stock.

    Tries vnstock_data (VCI) first, then falls back to Yahoo Finance.
    Symbol should be the raw ticker (e.g., VIC, HPG).
    """
    symbol = symbol.strip().upper()

    # 1. Try vnstock_data (paid API) — much better coverage for VN stocks
    if HAS_VNSTOCK_DATA:
        try:
            result = _extract_vnstock_data(symbol)
            if result is not None:
                logger.info(
                    "vnstock_data supplement for %s (period=%s, revenue=%s, equity=%s)",
                    symbol,
                    result.period,
                    result.revenue,
                    result.equity,
                )
                return result
        except Exception as exc:
            logger.warning(
                "vnstock_data failed for %s, falling back to yfinance: %s",
                symbol,
                exc,
            )

    # 2. Fallback to Yahoo Finance
    try:
        result = _extract_yfinance_data(symbol)
        logger.info(
            "yfinance fallback for %s (period=%s, revenue=%s)",
            symbol,
            result.period,
            result.revenue,
        )
        return result
    except Exception as exc:
        logger.error("Error fetching yfinance data for %s.VN: %s", symbol, exc)
        return YFinanceFinancialResponse(symbol=symbol, period="error")
