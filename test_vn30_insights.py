"""
Standalone VN30 AI Insight Test Script
========================================
Chạy riêng AI Insight cho 30 mã VN30, không lưu DB.
Xuất kết quả ra file Markdown + JSON để kiểm tra hôm sau.

Cách chạy:
    cd ai
    python test_vn30_insights.py

Yêu cầu:
    - Đã cài dependencies từ requirements.txt (vnstock, pandas, numpy)
    - Đã cấu hình .env với ít nhất 1 LLM API key (Gemini/OpenRouter/Blackbox/Beeknoee)
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix Windows console encoding for Vietnamese output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# Ensure ai/src is on path when running from ai/ directory
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.application.services.insight_service import InsightService
from src.api.dependencies import LLMProviderSelector
from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# VN30 Universe
# ---------------------------------------------------------------------------
VN30_SYMBOLS = [
    "ACB", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB",
    "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE"
]

# ---------------------------------------------------------------------------
# Data fetching helpers (vnstock v3)
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from vnstock. Returns last `days` rows."""
    try:
        from vnstock import Vnstock
        sources = ["KBS", "VCI", "MSN"]
        df = None
        last_err = None
        for src in sources:
            try:
                stock = Vnstock().stock(symbol=symbol, source=src)
                df = stock.quote.history(start=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                                         end=datetime.now().strftime("%Y-%m-%d"),
                                         interval="1D")
                if df is not None and not df.empty:
                    break
            except Exception as exc:
                last_err = exc
                continue
        if df is None or df.empty:
            if last_err:
                raise last_err
            return None
        if df is None or df.empty:
            return None
        df = df.copy()
        # Normalize column names (vnstock may return different cases)
        df.columns = [str(c).lower().strip() for c in df.columns]
        # Ensure required columns exist
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            # Try Vietnamese column names sometimes returned
            mapping = {
                "giá mở cửa": "open", "giá cao nhất": "high",
                "giá thấp nhất": "low", "giá đóng cửa": "close",
                "khối lượng": "volume"
            }
            for old, new in mapping.items():
                if old in df.columns and new not in df.columns:
                    df[new] = df[old]
        if not required.issubset(set(df.columns)):
            return None
        df = df[list(required.intersection(df.columns)) + [c for c in df.columns if c not in required]]
        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)
        else:
            df = df.sort_index().reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning(f"Failed to fetch OHLCV for {symbol}: {exc}")
        return None


def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """Return latest RSI value."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Return latest MACD, Signal, Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
    }


def calculate_sma(series: pd.Series, window: int = 20) -> float:
    return float(series.rolling(window=window).mean().iloc[-1])


def build_technical_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Build technical_data dict matching AI service expectation."""
    close = df["close"]
    volume = df["volume"]
    current_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price

    rsi = calculate_rsi(close)
    macd_vals = calculate_macd(close)
    sma20 = calculate_sma(close, 20)
    sma50 = calculate_sma(close, 50) if len(close) >= 50 else None

    avg_vol_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
    latest_vol = float(volume.iloc[-1])
    vol_ratio = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # Simple trend
    trend = "Bullish" if current_price > sma20 else "Bearish" if current_price < sma20 else "Neutral"

    period_high = float(df["high"].tail(48).max()) if len(df) >= 48 else float(df["high"].max())
    period_low = float(df["low"].tail(48).min()) if len(df) >= 48 else float(df["low"].min())
    change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

    return {
        "price": current_price,
        "previous_close": prev_close,
        "change_percent": round(change_pct, 2),
        "period_high": period_high,
        "period_low": period_low,
        "rsi": round(rsi, 2),
        "macd": f"MACD={round(macd_vals['macd'], 2)}, Signal={round(macd_vals['signal'], 2)}, Hist={round(macd_vals['histogram'], 2)}",
        "macd_histogram": round(macd_vals["histogram"], 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2) if sma50 else None,
        "volume": int(latest_vol),
        "volume_avg_20d": int(avg_vol_20),
        "volume_ratio_20d": round(vol_ratio, 2),
        "trend": trend,
        "bars_used": len(df),
    }


# ---------------------------------------------------------------------------
# Insight execution
# ---------------------------------------------------------------------------

async def generate_insight_for_symbol(
    insight_service: InsightService,
    symbol: str,
    mode: str = "strict"
) -> Optional[Dict[str, Any]]:
    """Fetch data and generate insight for a single symbol."""
    logger.info(f"Fetching data for {symbol} ...")
    df = fetch_ohlcv(symbol, days=120)
    if df is None or len(df) < 30:
        logger.warning(f"Insufficient data for {symbol}, skipping.")
        return None

    technical_data = build_technical_data(df)
    # Fundamental & sentiment are left empty for this standalone test
    # because fetching real fundamentals requires additional vnstock APIs
    # and sentiment requires news DB. The LLM will still generate insight
    # based on technical data.
    fundamental_data: Dict[str, Any] = {}
    sentiment_data: Dict[str, Any] = {
        "score": 0.0,
        "sentiment": "Trung lập (không có dữ liệu tin tức trong chế độ test)",
        "recent_news": "Không có tin tức"
    }

    try:
        result = await insight_service.generate_insight(
            symbol=symbol,
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            sentiment_data=sentiment_data,
            mode=mode,
        )
        result["generated_at"] = datetime.now().isoformat()
        result["input_data"] = {
            "technical": technical_data,
            "fundamental": fundamental_data,
            "sentiment": sentiment_data,
        }
        return result
    except Exception as exc:
        logger.error(f"Insight generation failed for {symbol}: {exc}")
        return {
            "symbol": symbol,
            "type": "Hold",
            "title": "Lỗi khi tạo insight",
            "description": str(exc),
            "confidence": 0,
            "reasoning": [f"Lỗi LLM/API: {exc}"],
            "target_price": None,
            "stop_loss": None,
            "evidence": [],
            "metadata": {"error": str(exc)},
            "generated_at": datetime.now().isoformat(),
            "input_data": {"technical": technical_data},
        }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Write a Markdown report for easy human review."""
    lines: List[str] = []
    lines.append("# VN30 AI Insight Test Report")
    lines.append("")
    lines.append(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Next check date:** {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}")
    lines.append(f"**Total symbols:** {len(results)}")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| # | Symbol | Type | Confidence | Current Price | Target Price | Stop Loss | Title |")
    lines.append("|---|--------|------|------------|---------------|--------------|-----------|-------|")

    for idx, r in enumerate(results, 1):
        sym = r.get("symbol", "N/A")
        typ = r.get("type", "N/A")
        conf = r.get("confidence", "N/A")
        tech = r.get("input_data", {}).get("technical", {})
        price = tech.get("price", "N/A")
        tgt = r.get("target_price") if r.get("target_price") is not None else "-"
        sl = r.get("stop_loss") if r.get("stop_loss") is not None else "-"
        title = r.get("title", "").replace("|", "/")
        lines.append(f"| {idx} | {sym} | {typ} | {conf} | {price} | {tgt} | {sl} | {title} |")

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    for r in results:
        sym = r.get("symbol", "N/A")
        lines.append(f"### {sym}")
        lines.append("")
        lines.append(f"- **Type:** {r.get('type', 'N/A')}")
        lines.append(f"- **Confidence:** {r.get('confidence', 'N/A')}")
        lines.append(f"- **Title:** {r.get('title', '')}")
        lines.append(f"- **Description:** {r.get('description', '')}")
        tgt = r.get("target_price")
        sl = r.get("stop_loss")
        lines.append(f"- **Target Price:** {tgt if tgt is not None else 'N/A'}")
        lines.append(f"- **Stop Loss:** {sl if sl is not None else 'N/A'}")
        lines.append("")
        lines.append("**Reasoning:**")
        for reason in r.get("reasoning", []):
            lines.append(f"- {reason}")
        lines.append("")
        lines.append("**Evidence:**")
        for ev in r.get("evidence", []):
            lines.append(f"- {ev}")
        lines.append("")
        tech = r.get("input_data", {}).get("technical", {})
        if tech:
            lines.append("**Input Technical Data:**")
            for k, v in tech.items():
                lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report saved to: {output_path}")


def generate_json_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Write full JSON report for programmatic inspection."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "next_check_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "total_symbols": len(results),
        "symbols": VN30_SYMBOLS,
        "results": results,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"JSON report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    settings = get_settings()
    selector = LLMProviderSelector(settings)
    # Force OpenRouter while Beeknoee is down (502/504)
    provider = selector.select_provider(force_provider="openrouter")
    insight_service = InsightService(llm_provider=provider)

    # Determine mode (default strict as in production when canary_ratio=1.0)
    mode = "strict"
    logger.info(f"Starting VN30 insight test — mode={mode}, provider={type(provider).__name__}")

    results: List[Dict[str, Any]] = []
    for symbol in VN30_SYMBOLS:
        result = await generate_insight_for_symbol(insight_service, symbol, mode=mode)
        if result:
            results.append(result)
        # Small delay to avoid rate limiting on free tiers
        await asyncio.sleep(0.5)

    # Ensure output directory exists
    output_dir = current_dir / "test_results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"vn30_insights_{timestamp}.md"
    json_path = output_dir / f"vn30_insights_{timestamp}.json"

    generate_markdown_report(results, md_path)
    generate_json_report(results, json_path)

    logger.info("Done! Check the test_results/ folder.")
    print(f"\n✅ Hoàn tất! Kết quả được lưu tại:")
    print(f"   - Markdown: {md_path}")
    print(f"   - JSON    : {json_path}")
    print(f"\n💡 Gợi ý: Ngày mai hãy so sánh giá đóng cửa thực tế với:")
    print(f"   - Target Price (nếu Buy) để kiểm tra độ chính xác")
    print(f"   - Stop Loss để kiểm tra rủi ro")


if __name__ == "__main__":
    asyncio.run(main())
