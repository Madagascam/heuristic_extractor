import os
from prompt_lib import Prompt
from verifier import ResponseVerifier
from g4f import ChatCompletion, Provider

# Словарь с моделями и провайдерами для тестирования.
providers_and_models = {
    "deepseek-r1": {"model": "deepseek-r1", "provider": Provider.Blackbox},
    "gpt-4o": {"model": "claude-3-haiku", "provider": Provider.DDG}
}

# Директория с промптами
PROMPTS_DIR = "prompts"

# Список типов промптов (имена папок внутри PROMPTS_DIR)
prompt_types = ["single", "single_with_score", "multiple", "multiple_with_score"]

# Пример PGN для тестирования
sample_pgn = (
    '[Event "Test Game"]\n'
    '[Site "Test Site"]\n'
    '[Date "2025.03.14"]\n'
    '[Round "1"]\n'
    '[White "Player1"]\n'
    '[Black "Player2"]\n'
    '[Result "*"]\n\n'
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6\n"
)

def test_prompts():
    for prompt_type in prompt_types:
        prompt_folder = os.path.join(PROMPTS_DIR, prompt_type)
        prompt_file_path = os.path.join(prompt_folder, "prompt.txt")
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except Exception as e:
            print(f"Не удалось открыть файл {prompt_file_path}: {e}")
            continue

        # Создаем верификатор для данного типа промпта.
        verifier = ResponseVerifier(expected_format=prompt_type)

        for key, config in providers_and_models.items():
            print(f"\nТестирование: Тип промпта '{prompt_type}', Модель '{key}'")
            prompt_instance = Prompt(
                prompt_text=prompt_text,
                model=config["model"],
                provider=config["provider"],
                verify_func=verifier.verify
            )

            try:
                result = prompt_instance.get_verified_response(additional_context=sample_pgn)
                print(f"Результат: {result}")
            except Exception as e:
                print(f"Ошибка: {e}")

if __name__ == "__main__":
    test_prompts()
