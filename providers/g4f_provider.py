import time
from g4f import ChatCompletion, Provider
from core.interfaces import LLMServiceProvider, ModelConfig, RetryConfig, LLMGenerationError

class G4FProvider(LLMServiceProvider):
    """LLM Service Provider implementation using g4f (Simplified retry logic)."""

    PROVIDER_MAP = {
         "Blackbox": Provider.Blackbox,
    }

    def generate_response(self, prompt: str, model_config: ModelConfig, retry_config: RetryConfig) -> str:
        """Generates response using g4f with a simple manual retry loop."""

        max_attempts = retry_config.get('max_attempts', 3)
        delay_seconds = retry_config.get('delay_seconds', 5)
        attempts = 0
        last_exception = None

        model_name = model_config.get("model_name", "deepseek-chat")
        provider_config = model_config.get("provider", "Blackbox")

        provider_instance = None
        if isinstance(provider_config, str):
            provider_instance = self.PROVIDER_MAP.get(provider_config)
            if provider_instance is None:
                raise LLMGenerationError(f"Unknown g4f provider name: '{provider_config}'. Available names: {list(self.PROVIDER_MAP.keys())}")
        elif isinstance(provider_config, type) and issubclass(provider_config, Provider):
             provider_instance = provider_config
        else:
             raise LLMGenerationError(f"Invalid 'provider' in model_config. Expected string name or g4f Provider class. Got: {type(provider_config)}")

        provider_name = provider_instance.__name__ if isinstance(provider_instance, type) else str(provider_instance)

        while attempts < max_attempts:
            attempts += 1
            print(f"--- Calling g4f (Attempt {attempts}/{max_attempts}): Model='{model_name}', Provider='{provider_name}' ---")
            try:
                response = ChatCompletion.create(
                    model=model_name,
                    provider=provider_instance,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )

                if not isinstance(response, str):
                    print(f"Warning: Received non-string response type from '{provider_name}' (Attempt {attempts}): {type(response)}. Retrying...")
                    last_exception = LLMGenerationError(f"Non-string response type: {type(response)}")
                    if attempts < max_attempts:
                         time.sleep(delay_seconds)
                    continue

                if not response.strip():
                    print(f"Warning: Received empty response from '{provider_name}' (Attempt {attempts}). Retrying...")
                    last_exception = LLMGenerationError("Empty response")
                    if attempts < max_attempts:
                         time.sleep(delay_seconds)
                    continue

                print(f"--- g4f call successful (Attempt {attempts}) ---")
                return response

            except Exception as e:
                print(f"g4f call failed (Attempt {attempts}/{max_attempts}) with error: {type(e).__name__}: {e}")
                last_exception = e
                if attempts < max_attempts:
                    print(f"Retrying in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)

        print(f"--- g4f call failed after {max_attempts} attempts. ---")
        raise LLMGenerationError(f"Failed to get valid response from g4f provider '{provider_name}' after {max_attempts} attempts. Last error: {type(last_exception).__name__}: {last_exception}") from last_exception