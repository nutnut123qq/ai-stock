"""POST /api/analyze — LangGraph dashboard forecast (StockAnalyst client on .NET)."""
import os
import re

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.langgraph_stock.graph import build_ta_graph

load_dotenv()

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")

router = APIRouter()


class AnalyzeRequest(BaseModel):
    symbol: str
    news_context: str = Field(
        default="",
        description="Optional override if backend news fetch fails or is empty",
    )
    tech_context: str = Field(
        default="",
        description="Optional override if backend technical summary fetch fails or is empty",
    )


def _normalize_symbol(raw: str) -> str:
    s = (raw or "").strip().upper()
    if not s:
        raise HTTPException(status_code=400, detail="symbol is required")
    if not _SYMBOL_RE.match(s):
        raise HTTPException(
            status_code=400,
            detail="symbol must be 2–20 alphanumeric characters (e.g. VNM, FPT).",
        )
    return s


def _build_llm():
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY not found in .env (required when LLM_PROVIDER=gemini).",
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
            max_retries=1,
            timeout=20,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "600"))
        num_ctx = os.getenv("OLLAMA_NUM_CTX", "").strip()
        kwargs: dict = {
            "model": model,
            "base_url": base_url,
            "temperature": 0.7,
            "sync_client_kwargs": {"timeout": timeout_s},
            "async_client_kwargs": {"timeout": timeout_s},
        }
        if num_ctx.isdigit():
            kwargs["num_ctx"] = int(num_ctx)
        return ChatOllama(**kwargs)

    if provider == "blackbox":
        api_key = os.getenv("BLACKBOX_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="BLACKBOX_API_KEY not found in .env (required when LLM_PROVIDER=blackbox).",
            )
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("BLACKBOX_BASE_URL", "https://api.blackbox.ai").rstrip("/")
        model = os.getenv("BLACKBOX_MODEL", "blackboxai/openai/gpt-5.2")
        timeout_s = float(os.getenv("BLACKBOX_TIMEOUT", "120"))
        return ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_retries=1,
            request_timeout=timeout_s,
        )

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENROUTER_API_KEY not found in .env (required when LLM_PROVIDER=openrouter).",
            )
        from langchain_openai import ChatOpenAI

        base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        timeout_s = float(os.getenv("OPENROUTER_TIMEOUT", "120"))
        return ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_retries=1,
            request_timeout=timeout_s,
        )

    raise HTTPException(
        status_code=500,
        detail=(
            f"Unknown LLM_PROVIDER={provider!r}. "
            "Use 'ollama', 'gemini', 'blackbox', or 'openrouter'."
        ),
    )


@router.post("/analyze")
async def analyze_stock(request: AnalyzeRequest):
    symbol = _normalize_symbol(request.symbol)

    llm = _build_llm()

    try:
        backend_base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:5197")

        graph = build_ta_graph(llm=llm, backend_base_url=backend_base_url)

        state = {
            "symbol": symbol,
            "news_context": request.news_context or "",
            "tech_context": request.tech_context or "",
        }

        return graph.invoke(state)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TA graph analysis failed: {e!s}",
        ) from e
