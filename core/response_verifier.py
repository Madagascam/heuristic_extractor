import json
from typing import Union, List, Dict, Any

from .interfaces import ResponseVerifierInterface, PromptParams, VerificationError


class ResponseVerifier(ResponseVerifierInterface):
    """Verifies the LLM response based on expected format derived from prompt_params."""

    def verify(self, response_string: str, prompt_params: PromptParams) -> Union[List, Dict]:
        """
        Verifies the response string.
        Raises VerificationError if validation fails.
        """
        response_string = response_string.strip()
        if response_string.startswith("```json"):
            response_string = response_string[7:]
        elif response_string.startswith("```"):
            response_string = response_string[3:]

        if response_string.endswith("```"):
            response_string = response_string[:-3]

        if response_string.startswith("JSON Output:"):
            response_string = response_string.split("JSON Output:", 1)[-1]

        response_string = response_string.strip()

        if not response_string:
            raise VerificationError("Received empty string after cleaning potential markdown/artifacts.")

        try:
            data = json.loads(response_string)
        except json.JSONDecodeError as e:
            raise VerificationError(f"Invalid JSON: {e}. Response was: '{response_string[:200]}...'") from e

        scope = prompt_params.get("scope", "multiple")
        output_format = prompt_params.get("output_format", "simple")
        # Получаем категории и диапазон очков из параметров
        categories = prompt_params.get("categories")
        score_range = prompt_params.get("score_range", (1, 10))

        if scope == "single":
            # Для single, сам 'data' должен быть одним сегментом (списком)
            self._verify_single_segment(data, output_format, scope, score_range, categories)
        elif scope == "multiple":
            # Для multiple, 'data' должен быть списком сегментов (списком списков)
            if not isinstance(data, list):
                raise VerificationError(f"Expected a list of segments for 'multiple' scope, got {type(data)}.")
            # Проверяем каждый сегмент в списке
            for segment in data:
                self._verify_single_segment(segment, output_format, scope, score_range, categories)
        else:
            raise VerificationError(f"Unknown scope specified in prompt_params: {scope}")

        return data

    def _verify_single_segment(self,
                               segment: Any,
                               output_format: str,
                               scope: str,
                               score_range: tuple,
                               categories: List[str] | None):
        """Verifies one segment based on the output format."""
        if not isinstance(segment, list):
            raise VerificationError(f"Expected each segment to be a list, got {type(segment)}: {segment}")

        # Определяем ожидаемую длину и типы на основе формата
        expected_len = 0
        expected_types = []
        if output_format == "simple":
            expected_len = 2
            expected_types = [int, int]
        elif output_format == "with_score":
            expected_len = 3
            expected_types = [int, int, int]
        elif output_format == "with_category":
            expected_len = 4
            expected_types = [int, int, int, str]
        elif output_format == "with_reason":
            expected_len = 3
            expected_types = [int, int, str]
        else:
            raise VerificationError(f"Unknown output_format specified for verification: {output_format}")

        if len(segment) != expected_len:
            raise VerificationError(
                f"Expected segment of length {expected_len} for format '{output_format}', got length {len(segment)} in segment: {segment}")

        # Проверка типов элементов и базовых ограничений
        for i, (item, expected_type) in enumerate(zip(segment, expected_types)):
            if expected_type == str:
                if not isinstance(item, str):
                    raise VerificationError(
                        f"Expected type {expected_type} (string) at index {i}, got type {type(item)} in segment: {segment}")
                if not item.strip():
                    field_name = "reason_string" if output_format == "with_reason" else "category"
                    raise VerificationError(f"Empty {field_name} found at index {i} in segment: {segment}")
            elif expected_type == int:
                if not isinstance(item, int):
                    raise VerificationError(
                        f"Expected type {expected_type} (integer) at index {i}, got type {type(item)} in segment: {segment}")
                if i < 2 and item < 0:
                    raise VerificationError(
                        f"Half-move index cannot be negative. Got {item} at index {i} in segment: {segment}")
            elif not isinstance(item, expected_type):
                raise VerificationError(
                    f"Expected type {expected_type} at index {i}, got type {type(item)} in segment: {segment}")

        if segment[0] > segment[1]:
            raise VerificationError(
                f"start_halfmove ({segment[0]}) cannot be greater than end_halfmove ({segment[1]}) in segment: {segment}")

        # Проверка диапазона score
        if output_format == "with_score" or output_format == "with_category":
            score = segment[2]
            if not (isinstance(score_range, tuple) and len(score_range) == 2 and
                    isinstance(score_range[0], (int, float)) and isinstance(score_range[1], (int, float))):
                raise VerificationError(
                    f"Invalid score_range configuration: {score_range}. Expected tuple of two numbers.")
            min_score, max_score = score_range
            if not (min_score <= score <= max_score):
                raise VerificationError(f"Score {score} out of configured range {score_range} in segment: {segment}")

        if output_format == "with_category":
            category = segment[3]
            if categories is None:
                raise VerificationError(
                    "Output format is 'with_category' but no 'categories' list was provided for verification.")
            if not isinstance(categories, list):
                raise VerificationError(f"Invalid categories configuration: {categories}. Expected a list of strings.")
            if category not in categories:
                raise VerificationError(
                    f"Invalid category '{category}'. Allowed categories: {categories} in segment: {segment}")
