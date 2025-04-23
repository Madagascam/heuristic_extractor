import json
import os
import time
from typing import List, Dict, Any

from core.highlight_extractor import HighlightExtractor
from core.interfaces import HighlightExtractionError, PromptParams, RetryConfig, ModelConfig
from core.pgn_parser import PGNParser
from core.prompt_generator import PromptGenerator
from core.response_verifier import ResponseVerifier
from providers.google_provider import GoogleProvider

# 1. Пути
PGN_INPUT_DIR = "pgns"
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "google_models_results.jsonl"

# 2. Google API Key
GOOGLE_API_KEY = ""

# 3. Модели Google для тестирования
GOOGLE_MODELS_TO_TEST: List[str] = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking-exp-01-21",

]

# 4. Параметры Промпта
COMMON_PROMPT_PARAMS: PromptParams = {
    'scope': 'multiple',
    'focus': 'youtube',
    'output_format': 'with_score',
    'use_few_shot': False,
}

# 5. Параметры Ретрая (общие для всех запусков)
COMMON_RETRY_CONFIG: RetryConfig = {'max_attempts': 2, 'delay_seconds': 5}

parser = PGNParser()
prompt_generator = PromptGenerator(prompt_dir="prompts/blocks")
response_verifier = ResponseVerifier()
google_provider = GoogleProvider(api_key=GOOGLE_API_KEY)

extractor = HighlightExtractor(
    parser=parser,
    prompt_gen=prompt_generator,
    llm_service=google_provider,
    verifier=response_verifier
)


def process_all_pgns():
    """
    Обрабатывает все PGN файлы в PGN_INPUT_DIR, используя все модели
    из GOOGLE_MODELS_TO_TEST, и сохраняет результаты в OUTPUT_FILENAME.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print(f"Starting PGN processing...")
    print(f"Input directory: {os.path.abspath(PGN_INPUT_DIR)}")
    print(f"Output file: {os.path.abspath(output_file_path)}")
    print(f"Models to test: {GOOGLE_MODELS_TO_TEST}")
    print(f"Prompt params: {COMMON_PROMPT_PARAMS}")

    # Счетчик обработанных файлов и результатов
    files_processed_count = 0
    results_saved_count = 0

    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        try:
            pgn_files = [f for f in os.listdir(PGN_INPUT_DIR) if f.lower().endswith(".pgn")]
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Input directory not found: {PGN_INPUT_DIR}")
            return
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to list files in input directory: {e}")
            return

        if not pgn_files:
            print(f"Warning: No .pgn files found in {PGN_INPUT_DIR}")
            return

        print(f"Found {len(pgn_files)} PGN files to process.")

        for filename in pgn_files:
            files_processed_count += 1
            pgn_file_path = os.path.join(PGN_INPUT_DIR, filename)
            print(f"\n[{files_processed_count}/{len(pgn_files)}] Processing file: {filename}")

            # Читаем содержимое PGN файла
            try:
                with open(pgn_file_path, 'r', encoding='utf-8') as f:
                    pgn_content = f.read()
            except Exception as e:
                print(f"  ERROR: Failed to read PGN file {filename}: {e}")
                error_result = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pgn_file": filename,
                    "model_name": None,
                    "prompt_params": COMMON_PROMPT_PARAMS,
                    "status": "error",
                    "result": f"Failed to read PGN file: {e}"
                }
                outfile.write(json.dumps(error_result) + "\n")
                results_saved_count += 1
                continue

            for model_name in GOOGLE_MODELS_TO_TEST:
                print(f"  Running model: {model_name}...")

                current_model_config: ModelConfig = {'model_name': model_name}

                result_data: Dict[str, Any] = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pgn_file": filename,
                    "model_name": model_name,
                    "prompt_params": COMMON_PROMPT_PARAMS,
                    "status": None,
                    "result": None
                }

                try:
                    highlights = extractor.extract(
                        pgn_string=pgn_content,
                        model_config=current_model_config,
                        prompt_params=COMMON_PROMPT_PARAMS,
                        retry_config=COMMON_RETRY_CONFIG
                    )
                    result_data["status"] = "success"
                    result_data["result"] = highlights
                    print(
                        f"    Success: Found {len(highlights) if isinstance(highlights, list) else 'N/A'} highlights.")

                except HighlightExtractionError as e:
                    result_data["status"] = "error"
                    result_data["result"] = str(e)
                    print(f"    ERROR during extraction: {e}")
                except Exception as e:
                    result_data["status"] = "error"
                    result_data["result"] = f"Unexpected error: {type(e).__name__}: {e}"
                    print(f"    UNEXPECTED ERROR: {type(e).__name__}: {e}")

                try:
                    outfile.write(json.dumps(result_data) + "\n")
                    results_saved_count += 1
                except Exception as write_e:
                    print(f"  CRITICAL ERROR: Failed to write result to output file: {write_e}")

    print(f"\n--- Processing Finished ---")
    print(f"Total PGN files processed: {files_processed_count}")
    print(f"Total results saved (including errors): {results_saved_count}")
    print(f"Results saved to: {os.path.abspath(output_file_path)}")


if __name__ == "__main__":
    process_all_pgns()
