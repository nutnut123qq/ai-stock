import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

if "qdrant_client" not in sys.modules:
    qdrant_module = types.ModuleType("qdrant_client")
    qdrant_module.QdrantClient = object
    sys.modules["qdrant_client"] = qdrant_module

if "qdrant_client.http" not in sys.modules:
    sys.modules["qdrant_client.http"] = types.ModuleType("qdrant_client.http")

if "qdrant_client.http.exceptions" not in sys.modules:
    exceptions_module = types.ModuleType("qdrant_client.http.exceptions")
    exceptions_module.UnexpectedResponse = Exception
    sys.modules["qdrant_client.http.exceptions"] = exceptions_module

if "qdrant_client.http.models" not in sys.modules:
    models_module = types.ModuleType("qdrant_client.http.models")
    for model_name in ["Distance", "VectorParams", "Filter", "FieldCondition", "MatchValue", "PointStruct"]:
        setattr(models_module, model_name, object)
    sys.modules["qdrant_client.http.models"] = models_module

if "vnstock" not in sys.modules:
    vnstock_module = types.ModuleType("vnstock")
    vnstock_module.Vnstock = object
    vnstock_module.Listing = object
    sys.modules["vnstock"] = vnstock_module

from src.api.routes.forecast import ForecastRequest, generate_forecast
from src.api.routes.qa import QARequestV2, answer_question
from src.api.routes.rag import validate_api_key


class AiApiBehaviorTests(unittest.TestCase):
    def test_validate_api_key_missing_header_returns_401(self) -> None:
        settings = SimpleNamespace(internal_api_key="secret-key")
        with patch("src.api.routes.rag.get_settings", return_value=settings), patch(
            "src.api.routes.rag.os.getenv", return_value=None
        ):
            with self.assertRaises(HTTPException) as ctx:
                validate_api_key(None)

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Missing X-Internal-Api-Key", str(ctx.exception.detail))

    def test_validate_api_key_invalid_header_returns_401(self) -> None:
        settings = SimpleNamespace(internal_api_key="secret-key")
        with patch("src.api.routes.rag.get_settings", return_value=settings), patch(
            "src.api.routes.rag.os.getenv", return_value=None
        ):
            with self.assertRaises(HTTPException) as ctx:
                validate_api_key("wrong-key")

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid X-Internal-Api-Key", str(ctx.exception.detail))

    def test_validate_api_key_valid_header_returns_true(self) -> None:
        settings = SimpleNamespace(internal_api_key="secret-key")
        with patch("src.api.routes.rag.get_settings", return_value=settings), patch(
            "src.api.routes.rag.os.getenv", return_value=None
        ):
            is_valid = validate_api_key("secret-key")

        self.assertTrue(is_valid)


class AiRouteBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_forecast_route_returns_generated_timestamp(self) -> None:
        class UseCaseStub:
            async def execute(self, **kwargs):
                return {
                    "symbol": kwargs["symbol"],
                    "trend": "Up",
                    "confidence": "High",
                    "confidence_score": 0.88,
                    "time_horizon": kwargs["time_horizon"],
                    "recommendation": "Buy",
                    "key_drivers": ["driver"],
                    "risks": ["risk"],
                    "analysis": "analysis",
                }

        result = await generate_forecast(
            ForecastRequest(symbol="VNM", time_horizon="short"),
            use_case=UseCaseStub(),
        )

        self.assertEqual(result.symbol, "VNM")
        self.assertEqual(result.time_horizon, "short")
        self.assertTrue(result.generated_at)

    async def test_qa_route_wraps_service_error_to_503(self) -> None:
        class QaServiceStub:
            async def answer_question(self, **kwargs):
                raise RuntimeError("provider down")

        request = QARequestV2(question="What happened?")
        with self.assertRaises(HTTPException) as ctx:
            await answer_question(request, qa_service=QaServiceStub())

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("QA/LLM failed", str(ctx.exception.detail))

    async def test_qa_route_maps_sources_response(self) -> None:
        class QaServiceStub:
            async def answer_question(self, **kwargs):
                return {
                    "answer": "sample answer",
                    "sources": [
                        {
                            "documentId": "doc-1",
                            "source": "analysis_report",
                            "title": "Title",
                            "section": "Sec",
                            "symbol": "VNM",
                            "chunkId": "c1",
                            "score": 0.9,
                            "textPreview": "preview",
                        }
                    ],
                }

        request = QARequestV2(question="What happened?")
        response = await answer_question(request, qa_service=QaServiceStub())

        self.assertEqual(response.answer, "sample answer")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].documentId, "doc-1")


if __name__ == "__main__":
    unittest.main()
