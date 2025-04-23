import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from core.interfaces import LLMServiceProvider, ModelConfig, RetryConfig, LLMGenerationError

retry_strategy = retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type((
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        TimeoutError
    )),
    reraise=True
)


class GoogleProvider(LLMServiceProvider):
    """LLM Service Provider implementation using Google Generative AI."""

    def __init__(self, api_key: str | None = None):
        if api_key:
            genai.configure(api_key=api_key)
        try:
            genai.list_models()
            print("Google GenAI configured successfully.")
        except Exception as e:
            print(f"Warning: Google GenAI might not be configured correctly: {e}")

    @retry_strategy
    def generate_response(self, prompt: str, model_config: ModelConfig, retry_config: RetryConfig) -> str:
        """Generates response using Google GenAI with retries."""

        if retry_config:
            current_retry_strategy = retry(
                stop=stop_after_attempt(retry_config.get('max_attempts', 3)),
                wait=wait_fixed(retry_config.get('delay_seconds', 5)),
                retry=retry_if_exception_type((
                    google_exceptions.ResourceExhausted,
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.DeadlineExceeded,
                    TimeoutError
                )),
                reraise=True
            )
            return current_retry_strategy(self._generate_google)(prompt, model_config)
        else:
            return self._generate_google(prompt, model_config)

    def _generate_google(self, prompt: str, model_config: ModelConfig) -> str:
        """Internal method to call Google GenAI, decorated by tenacity."""
        model_name = model_config.get("model_name", "gemini-2.0-flash")

        print(f"--- Calling Google GenAI: Model='{model_name}' ---")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)

            if not response.parts:
                if response.prompt_feedback.block_reason:
                    raise LLMGenerationError(f"Google API blocked prompt: {response.prompt_feedback.block_reason}")
                elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                    raise LLMGenerationError(f"Google API finished with reason: {response.candidates[0].finish_reason}")
                else:
                    print("Warning: Received empty response from Google API.")
                    return ""

            return response.text

        except (
        google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded,
        TimeoutError) as e:
            print(f"Google GenAI attempt failed: {e}")
            raise
        except Exception as e:
            raise LLMGenerationError(f"Unexpected Google GenAI error: {e}") from e
