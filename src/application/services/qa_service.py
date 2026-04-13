"""QA service for answering questions with RAG."""
from typing import Dict, Any, Optional, List
from src.domain.interfaces.llm_provider import LLMProvider
from src.domain.interfaces.vector_store import VectorStore
from src.domain.interfaces.embedding_provider import EmbeddingProvider
from src.shared.logging import get_logger

logger = get_logger(__name__)


class QAService:
    """Service for answering questions using RAG (Retrieval-Augmented Generation)."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        embedding_service: EmbeddingProvider
    ):
        """
        Initialize QA service.
        
        Args:
            llm_provider: LLM provider for generating answers
            vector_store: Vector store for document retrieval
            embedding_service: Service for generating embeddings
        """
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        logger.info("Initialized QAService")

    async def answer_question(
        self,
        question: str,
        base_context: Optional[str] = None,
        top_k: int = 6,
        document_id: Optional[str] = None,
        source: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG with optional filters.
        
        Args:
            question: The question to answer
            base_context: Base context from caller (optional)
            top_k: Number of chunks to retrieve
            document_id: Filter by document ID (for report-specific Q&A)
            source: Filter by source type
            symbol: Filter by symbol
            
        Returns:
            Dictionary with answer and sources (list of objects)
        """
        logger.info(
            f"Answering question: {question[:100]}... "
            f"(filters: documentId={document_id}, source={source}, symbol={symbol})"
        )
        
        # Build filters for vector search
        filters = {
            "document_id": document_id,
            "source": source,
            "symbol": symbol
        }
        filters = {key: value for key, value in filters.items() if value is not None}

        # Callers (financial_report, news) send full text in base_context; skip Qdrant/embed search to save latency.
        skip_vector_search = (
            source in ("financial_report", "news")
            and base_context is not None
            and base_context.strip() != ""
        )

        sources_raw: List[Dict[str, Any]] = []
        if skip_vector_search:
            logger.debug("Skipping vector search for %s (base_context provided)", source)
        else:
            try:
                sources_raw = await self.vector_store.search(
                    query_text=question,
                    top_k=top_k,
                    filters=filters or None
                )
                logger.debug(f"Retrieved {len(sources_raw)} relevant chunks")
            except Exception as e:
                logger.warning(f"Vector store unavailable, fallback to base context only: {str(e)}")
                sources_raw = []

        # Normalize sources with safe fallbacks and request values
        sources_sorted = sorted(
            sources_raw,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True
        )[:top_k]

        sources: List[Dict[str, Any]] = []
        for hit in sources_sorted:
            text = (hit.get("text") or "").strip()
            text_preview = text[:350] if len(text) > 350 else text
            source_obj = {
                "documentId": hit.get("documentId") or document_id or "",
                "source": hit.get("source") or source or "",
                "sourceUrl": hit.get("sourceUrl") if hit.get("sourceUrl") else None,
                "title": hit.get("title") or "Unknown",
                "section": hit.get("section") or "",
                "symbol": hit.get("symbol") or symbol or "",
                "chunkId": hit.get("chunkId") or "",
                "score": float(hit.get("score") or 0.0),
                "textPreview": text_preview,
                "text": text
            }
            sources.append(source_obj)

        # Build prompt with base context + retrieved chunks
        prompt_parts = [
            "Bạn là trợ lý phân tích tài chính.",
            "Yêu cầu: trả lời ngắn gọn, bằng tiếng Việt.",
            "Chỉ sử dụng thông tin trong ngữ cảnh cung cấp, không bịa nguồn.",
            "Nếu không đủ thông tin, hãy nói rõ.",
            ""
        ]

        if base_context:
            prompt_parts.append("Ngữ cảnh cơ bản:")
            prompt_parts.append(base_context)
            prompt_parts.append("")

        if sources:
            prompt_parts.append("Ngữ cảnh từ tài liệu:")
            for idx, source_obj in enumerate(sources, 1):
                title = source_obj.get("title") or "Unknown"
                section = source_obj.get("section") or "N/A"
                source_url = source_obj.get("sourceUrl") or "null"
                prompt_parts.append(f"[{idx}] {title} - {section} ({source_url})")
                prompt_parts.append(source_obj.get("text") or source_obj.get("textPreview") or "")
                prompt_parts.append("")

        prompt_parts.append(f"Câu hỏi: {question}")
        prompt_parts.append("")
        prompt_parts.append("Trả lời:")

        prompt = "\n".join(prompt_parts)

        # Generate answer
        answer = await self.llm_provider.generate(prompt)
        logger.info("Successfully generated answer")

        # Strip internal text field before returning
        response_sources = []
        for source_obj in sources:
            trimmed_obj = {key: value for key, value in source_obj.items() if key != "text"}
            response_sources.append(trimmed_obj)

        return {
            "answer": answer,
            "sources": response_sources
        }

    async def analyze_financial_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze financial metrics and provide insights.
        
        Args:
            financial_data: Financial data dictionary
            
        Returns:
            Dictionary with analysis and metrics
        """
        logger.info("Analyzing financial metrics")
        
        prompt = f"""Phân tích các chỉ số tài chính sau và đưa ra đánh giá:

Dữ liệu tài chính:
{financial_data}

Hãy phân tích:
1. Tình hình tài chính tổng quan
2. Các chỉ số quan trọng (ROE, ROA, EPS, etc.)
3. Xu hướng tăng trưởng
4. Điểm mạnh và điểm yếu
5. Khuyến nghị cho nhà đầu tư

Trả lời bằng tiếng Việt, có cấu trúc rõ ràng."""

        try:
            analysis = await self.llm_provider.generate(prompt)
            logger.info("Successfully analyzed financial metrics")

            return {
                "analysis": analysis,
                "metrics": financial_data
            }
        except Exception as e:
            logger.error(f"Error analyzing financial metrics: {str(e)}")
            raise
