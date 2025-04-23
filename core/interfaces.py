from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any

ModelConfig = Dict[str, Any]
PromptParams = Dict[str, Any]
RetryConfig = Dict[str, Any]


class LLMServiceProvider(ABC):
    """Interface for LLM service providers."""

    @abstractmethod
    def generate_response(self, prompt: str, model_config: ModelConfig, retry_config: RetryConfig) -> str:
        """Generates a response from the LLM."""
        pass


class PGNParserInterface(ABC):
    """Interface for PGN parsers."""

    @abstractmethod
    def parse(self, pgn_string: str) -> str:
        """Parses PGN string into a formatted list of moves."""
        pass


class PromptGeneratorInterface(ABC):
    """Interface for prompt generators."""

    @abstractmethod
    def generate(self, prompt_params: PromptParams, game_moves: str) -> str:
        """Generates the final prompt string."""
        pass


class ResponseVerifierInterface(ABC):
    """Interface for response verifiers."""

    @abstractmethod
    def verify(self, response_string: str, prompt_params: PromptParams) -> Union[List, Dict]:
        """Verifies the LLM response string against expected format."""
        pass


class HighlightExtractionError(Exception):
    """Base exception for errors during highlight extraction."""
    pass


class LLMGenerationError(HighlightExtractionError):
    """Error during LLM response generation."""
    pass


class VerificationError(HighlightExtractionError):
    """Error during response verification."""
    pass


class PGNParsingError(HighlightExtractionError):
    """Error during PGN parsing."""
    pass
