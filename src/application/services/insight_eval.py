"""Offline evaluation for AI Insight core quality."""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.application.services.insight_service import InsightService
from src.shared.config import get_settings


@dataclass
class EvalCase:
    symbol: str
    expected_type: str
    technical_data: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]


def _load_cases(dataset_path: Path) -> List[EvalCase]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases: List[EvalCase] = []
    for item in raw:
        cases.append(
            EvalCase(
                symbol=item["symbol"],
                expected_type=item["expected_type"],
                technical_data=item.get("technical_data", {}),
                fundamental_data=item.get("fundamental_data", {}),
                sentiment_data=item.get("sentiment_data", {}),
            )
        )
    return cases


def _schema_ok(result: Dict[str, Any]) -> bool:
    required = {
        "symbol",
        "type",
        "title",
        "description",
        "confidence",
        "reasoning",
        "target_price",
        "stop_loss",
        "evidence",
        "metadata",
    }
    return required.issubset(set(result.keys()))


async def _run_eval(dataset_path: Path, mode: str) -> Dict[str, Any]:
    settings = get_settings()
    if settings.gemini_api_key:
        from src.infrastructure.llm.gemini_client import GeminiClient
        llm_provider = GeminiClient()
    elif settings.openrouter_api_key:
        from src.infrastructure.llm.openrouter_client import OpenRouterClient
        llm_provider = OpenRouterClient(model_name=settings.openrouter_model)
    else:
        from src.infrastructure.llm.blackbox_client import BlackboxClient
        llm_provider = BlackboxClient(model_name=settings.blackbox_model)

    service = InsightService(llm_provider)
    cases = _load_cases(dataset_path)

    total = len(cases)
    correct = 0
    schema_ok_count = 0
    invalid_count = 0
    calibration_error_sum = 0.0
    rows: List[Dict[str, Any]] = []

    for case in cases:
        result = await service.generate_insight(
            symbol=case.symbol,
            technical_data=case.technical_data,
            fundamental_data=case.fundamental_data,
            sentiment_data=case.sentiment_data,
            mode=mode,
        )
        prediction = result.get("type", "Hold")
        confidence = int(result.get("confidence", 0))
        match = prediction.lower() == case.expected_type.lower()
        if match:
            correct += 1
        schema_ok = _schema_ok(result)
        if schema_ok:
            schema_ok_count += 1
        else:
            invalid_count += 1

        calibration_error_sum += abs((confidence / 100.0) - (1.0 if match else 0.0))
        rows.append(
            {
                "symbol": case.symbol,
                "expected_type": case.expected_type,
                "predicted_type": prediction,
                "confidence": confidence,
                "schema_ok": schema_ok,
            }
        )

    return {
        "mode": mode,
        "total_cases": total,
        "classification_accuracy": round((correct / total) * 100, 2) if total else 0.0,
        "schema_compliance_rate": round((schema_ok_count / total) * 100, 2) if total else 0.0,
        "invalid_output_rate": round((invalid_count / total) * 100, 2) if total else 0.0,
        "confidence_calibration_error": round((calibration_error_sum / total), 4) if total else 0.0,
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI Insight core quality.")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset file")
    parser.add_argument("--mode", default="strict", choices=["strict", "legacy"], help="Prompt mode to evaluate")
    parser.add_argument("--output", default="", help="Optional output JSON file path")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    result = asyncio.run(_run_eval(dataset_path, args.mode))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
