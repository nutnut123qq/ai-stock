"""
Evaluate VN30 AI Insight accuracy after T+1 (28/04/2026 vs 27/04/2026).
Reads the previous JSON report, fetches T+1 closing prices, and produces an evaluation report.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure ai/src is on path
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def fetch_close_price(symbol: str) -> Optional[float]:
    """Fetch latest daily close from vnstock (KBS source)."""
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source="KBS")
        df = stock.quote.history(
            start=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"),
            interval="1D"
        )
        if df is None or df.empty:
            return None
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)
        else:
            df = df.sort_index().reset_index(drop=True)
        return float(df["close"].iloc[-1])
    except Exception as exc:
        print(f"[WARN] Failed to fetch {symbol}: {exc}")
        return None


def evaluate_signal(
    insight_type: str,
    current_price: float,
    t1_close: float,
    target_price: Optional[float],
    stop_loss: Optional[float]
) -> str:
    """Evaluate if the T+1 result validates the insight."""
    if insight_type == "Buy":
        if target_price and t1_close >= target_price:
            return "Đúng (vượt target)"
        if stop_loss and t1_close <= stop_loss:
            return "Sai (chạm stop loss)"
        if t1_close >= current_price:
            return "Đúng (tăng/ngang)"
        return "Chưa rõ (giảm nhẹ)"

    if insight_type == "Sell":
        if stop_loss and t1_close >= stop_loss:
            return "Sai (vượt stop loss)"
        if t1_close < current_price:
            return "Đúng (giảm)"
        if t1_close == current_price:
            return "Đúng (đứng giá)"
        return "Sai (tăng)"

    # Hold: check if price stayed within ±2.5%
    change_pct = (t1_close - current_price) / current_price * 100
    if abs(change_pct) <= 2.5:
        return "Đúng (biên độ hẹp)"
    if abs(change_pct) <= 4.0:
        return "Chưa rõ"
    return "Sai (biến động mạnh)"


def main() -> None:
    # Find the latest insight JSON report
    results_dir = current_dir / "test_results"
    json_files = sorted(results_dir.glob("vn30_insights_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        print("❌ No insight JSON report found in test_results/")
        sys.exit(1)

    latest_json = json_files[0]
    print(f"📂 Using insight report: {latest_json.name}")

    with open(latest_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    results: List[Dict[str, Any]] = []
    for item in report.get("results", []):
        symbol = item["symbol"]
        current_price = item.get("input_data", {}).get("technical", {}).get("price")
        insight_type = item.get("type", "Hold")
        target = item.get("target_price")
        stop = item.get("stop_loss")
        confidence = item.get("confidence", 0)

        t1_close = fetch_close_price(symbol)
        if t1_close is None or current_price is None:
            eval_result = "Lỗi dữ liệu"
            change_pct = None
        else:
            change_pct = round((t1_close - current_price) / current_price * 100, 2)
            eval_result = evaluate_signal(insight_type, current_price, t1_close, target, stop)

        results.append({
            "symbol": symbol,
            "type": insight_type,
            "confidence": confidence,
            "price_t0": current_price,
            "price_t1": t1_close,
            "change_pct": change_pct,
            "target": target,
            "stop_loss": stop,
            "result": eval_result,
        })

    # Generate Markdown report
    lines = []
    lines.append("# VN30 AI Insight T+1 Evaluation Report")
    lines.append("")
    lines.append(f"**Insight generated:** {report.get('generated_at', 'N/A')}")
    lines.append(f"**Evaluation date:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Total symbols:** {len(results)}")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Symbol | Type | Conf | Giá T0 | Giá T+1 | Thay đổi | Target | Stop Loss | Kết quả |")
    lines.append("|--------|------|------|--------|---------|----------|--------|-----------|---------|")

    correct = 0
    wrong = 0
    unclear = 0

    for r in results:
        sym = r["symbol"]
        typ = r["type"]
        conf = r["confidence"]
        p0 = r["price_t0"] if r["price_t0"] is not None else "N/A"
        p1 = r["price_t1"] if r["price_t1"] is not None else "N/A"
        chg = f"{r['change_pct']}%" if r["change_pct"] is not None else "N/A"
        tgt = r["target"] if r["target"] is not None else "-"
        sl = r["stop_loss"] if r["stop_loss"] is not None else "-"
        res = r["result"]

        if "Đúng" in res:
            correct += 1
        elif "Sai" in res:
            wrong += 1
        else:
            unclear += 1

        lines.append(f"| {sym} | {typ} | {conf} | {p0} | {p1} | {chg} | {tgt} | {sl} | {res} |")

    total_judged = correct + wrong
    accuracy = round(correct / total_judged * 100, 1) if total_judged > 0 else 0

    lines.append("")
    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- **Đúng:** {correct}")
    lines.append(f"- **Sai:** {wrong}")
    lines.append(f"- **Chưa rõ / Lỗi:** {unclear}")
    lines.append(f"- **Accuracy (Đúng / (Đúng+Sai)):** {accuracy}%")
    lines.append("")

    # Breakdown by signal type
    buy_results = [r for r in results if r["type"] == "Buy"]
    sell_results = [r for r in results if r["type"] == "Sell"]
    hold_results = [r for r in results if r["type"] == "Hold"]

    if buy_results:
        lines.append("### Buy signals")
        for r in buy_results:
            lines.append(f"- **{r['symbol']}**: Giá T0={r['price_t0']} → T1={r['price_t1']} ({r['change_pct']}%) → {r['result']}")
        lines.append("")

    if sell_results:
        lines.append("### Sell signals")
        for r in sell_results:
            lines.append(f"- **{r['symbol']}**: Giá T0={r['price_t0']} → T1={r['price_t1']} ({r['change_pct']}%) → {r['result']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Ghi chú:*")
    lines.append("- **Buy Đúng**: Giá T+1 ≥ Giá T0 hoặc vượt target")
    lines.append("- **Buy Sai**: Giá T+1 ≤ Stop loss")
    lines.append("- **Sell Đúng**: Giá T+1 < Giá T0")
    lines.append("- **Sell Sai**: Giá T+1 ≥ Stop loss")
    lines.append("- **Hold Đúng**: Biến động trong ±2.5%")
    lines.append("- **Hold Sai**: Biến động > 4%")

    output_path = results_dir / f"vn30_evaluation_t1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Evaluation report saved to: {output_path}")
    print(f"📊 Accuracy: {accuracy}% ({correct}/{total_judged})")


if __name__ == "__main__":
    main()
