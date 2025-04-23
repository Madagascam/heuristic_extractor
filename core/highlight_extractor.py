from typing import Union, List, Dict

from .interfaces import (
    PGNParserInterface, PromptGeneratorInterface, LLMServiceProvider,
    ResponseVerifierInterface, HighlightExtractionError, PGNParsingError,
    LLMGenerationError, VerificationError, ModelConfig, PromptParams, RetryConfig
)


class HighlightExtractor:
    """Orchestrates the highlight extraction process."""

    def __init__(self,
                 parser: PGNParserInterface,
                 prompt_gen: PromptGeneratorInterface,
                 llm_service: LLMServiceProvider,  # Inject specific provider instance
                 verifier: ResponseVerifierInterface):
        self.parser = parser
        self.prompt_generator = prompt_gen
        self.llm_service = llm_service  # Store the injected provider
        self.verifier = verifier

    def extract(self,
                pgn_string: str,
                model_config: ModelConfig,
                prompt_params: PromptParams,
                retry_config: RetryConfig) -> Union[List, Dict]:  # Return verified data or raise error
        """
        Extracts highlights from a PGN string.

        Raises:
            HighlightExtractionError (or subclasses) if any step fails.
        """
        print("\n--- Starting Highlight Extraction ---")

        # 1. Parse PGN
        try:
            print("Parsing PGN...")
            game_moves_str = self.parser.parse(pgn_string)
        except PGNParsingError as e:
            print(f"Error parsing PGN: {e}")
            raise

        # 2. Generate Prompt
        try:
            print("Generating prompt...")
            prompt = self.prompt_generator.generate(prompt_params, game_moves_str)
            print("-" * 20 + " GENERATED PROMPT " + "-" * 20)
            print(prompt)
            print("-" * (40 + len(" GENERATED PROMPT ")))
        except Exception as e:
            print(f"Error generating prompt: {e}")
            raise HighlightExtractionError(f"Prompt generation failed: {e}") from e

        # 3. Call LLM Service (using the injected provider)
        try:
            print(f"Sending request to LLM service ({self.llm_service.__class__.__name__})...")
            raw_response = self.llm_service.generate_response(prompt, model_config, retry_config)
            print(f"Received raw response (first 200 chars): {raw_response[:200]}...")
        except LLMGenerationError as e:
            print(f"Error generating response from LLM: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error from LLM provider: {e}")
            raise LLMGenerationError(f"Unexpected provider error: {e}") from e

        # 4. Verify Response
        try:
            print("Verifying response...")
            verified_data = self.verifier.verify(raw_response, prompt_params)
            print("Verification successful.")
            return verified_data
        except VerificationError as e:
            print(f"Error verifying response: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during verification: {e}")
            raise VerificationError(f"Unexpected verification error: {e}") from e
