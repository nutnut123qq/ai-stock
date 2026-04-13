"""Use case for parsing natural language alerts."""
from src.application.services.nlp_parser_service import NLPParserService
from src.shared.logging import get_logger

logger = get_logger(__name__)


class ParseAlertUseCase:
    """Use case for parsing natural language alert requests."""
    
    def __init__(self, nlp_parser_service: NLPParserService):
        """
        Initialize parse alert use case.
        
        Args:
            nlp_parser_service: NLP parser service for parsing alerts
        """
        self.nlp_parser_service = nlp_parser_service
        logger.info("Initialized ParseAlertUseCase")

    async def execute(self, natural_language_input: str) -> dict:
        """
        Execute alert parsing use case.
        
        Args:
            natural_language_input: Natural language alert request
            
        Returns:
            Parsed alert dictionary
        """
        logger.info(f"Executing alert parsing: {natural_language_input[:100]}...")
        return await self.nlp_parser_service.parse_alert_intent(natural_language_input)
