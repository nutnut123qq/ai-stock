"""Use case for answering questions with RAG."""
from src.application.services.qa_service import QAService
from src.shared.logging import get_logger

logger = get_logger(__name__)


class AnswerQuestionUseCase:
    """Use case for answering questions using RAG."""
    
    def __init__(self, qa_service: QAService):
        """
        Initialize answer question use case.
        
        Args:
            qa_service: QA service for answering questions
        """
        self.qa_service = qa_service
        logger.info("Initialized AnswerQuestionUseCase")

    async def execute(self, question: str, base_context: str) -> dict:
        """
        Execute question answering use case.
        
        Args:
            question: The question to answer
            base_context: Additional context
            
        Returns:
            Answer dictionary
        """
        logger.info(f"Executing question answering: {question[:100]}...")
        return await self.qa_service.answer_question(question, base_context)
