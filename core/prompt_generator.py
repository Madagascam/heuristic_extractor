import os
from typing import Dict, List

from .interfaces import PromptGeneratorInterface, PromptParams


class PromptGenerator(PromptGeneratorInterface):
    """Generates prompts by assembling text blocks."""

    def __init__(self, prompt_dir: str = "prompts/blocks"):
        """
        Initializes the PromptGenerator.

        Args:
            prompt_dir: Path to the directory containing prompt block .txt files.
        """
        self.prompt_dir = prompt_dir
        # Загружаем блоки при создании экземпляра
        self.blocks = self._load_prompt_blocks()
        if not self.blocks:
            print(
                f"CRITICAL WARNING: No prompt blocks loaded from '{self.prompt_dir}'. Prompt generation will likely fail.")

    def _load_prompt_blocks(self) -> Dict[str, str]:
        """Loads text content from files in the prompt directory."""
        blocks = {}
        if not os.path.isdir(self.prompt_dir):
            print(f"Error: Prompt directory '{self.prompt_dir}' not found or is not a directory.")
            return blocks
        try:
            print(f"Loading prompt blocks from: {os.path.abspath(self.prompt_dir)}")
            found_files = []
            for filename in os.listdir(self.prompt_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(self.prompt_dir, filename)
                    block_name = filename[:-4]  # .txt
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            blocks[block_name] = f.read().strip()
                        found_files.append(filename)
                    except Exception as file_e:
                        print(f"Error reading prompt block file '{filepath}': {file_e}")
            if found_files:
                print(f"Successfully loaded prompt blocks: {', '.join(found_files)}")
            else:
                print(f"Warning: No .txt files found in prompt directory '{self.prompt_dir}'.")

        except Exception as e:
            print(f"Error listing files in prompt directory '{self.prompt_dir}': {e}")
        return blocks

    def generate(self, prompt_params: PromptParams, game_moves: str) -> str:
        """
        Assembles the final prompt string based on parameters.

        Args:
            prompt_params: Dictionary containing parameters like 'scope', 'focus', 'output_format', 'categories'.
            game_moves: A string containing the formatted list of game moves.

        Returns:
            The fully assembled prompt string.

        Raises:
             ValueError: If essential prompt blocks are missing (currently commented out).
        """
        parts = []
        missing_blocks = []

        def add_block(key: str):
            content = self.blocks.get(key)
            if content is not None:
                parts.append(content)
            else:
                missing_blocks.append(key)
                parts.append(f"[[ERROR: Prompt block '{key}' not found]]")

        # 1. Role/Base Task
        add_block("role_task_base")

        # 2. Task Modifier (Single/Multiple)
        scope = prompt_params.get("scope", "multiple")
        modifier_key = f"task_modifier_{scope}"
        add_block(modifier_key)

        # 3. Focus
        focus = prompt_params.get("focus", "youtube")
        focus_key = f"focus_{focus}"
        add_block(focus_key)

        # 4. Input Format Description
        add_block("input_format")

        # 5. Output Format Description (Используем разделенные блоки)
        output_format = prompt_params.get("output_format", "simple")
        # Формируем ключ блока на основе И формата И скоупа
        output_key = f"output_{output_format}_{scope}"
        output_text_template = self.blocks.get(output_key)

        if output_text_template is not None:
            final_output_text = output_text_template
            if output_format == "with_category":
                categories: List[str] = prompt_params.get("categories", [])
                if not categories:
                    print(
                        f"Warning: output_format is 'with_category' but no 'categories' list provided in prompt_params.")
                    categories_str = "[NO CATEGORIES SPECIFIED]"
                else:
                    categories_str = ", ".join([f"'{cat}'" for cat in categories])
                final_output_text = output_text_template.replace("{categories}", categories_str)
            parts.append(final_output_text)
        else:
            missing_blocks.append(output_key)
            parts.append(f"[[ERROR: Prompt block '{output_key}' not found]]")

        # 6. Few-Shot Examples
        if prompt_params.get("use_few_shot", False):
            example_key = f"example_{scope}_{focus}_{output_format}"
            add_block(example_key)

        # 7. Game Data - добавляем всегда, даже если были ошибки выше
        if not game_moves:
            print("Warning: Generating prompt with empty game_moves string.")
            parts.append("List of moves:\n[No moves provided]")
        else:
            parts.append(f"List of moves:\n{game_moves}")

        parts.append("\nJSON Output:")

        # Проверяем, были ли пропущены блоки
        if missing_blocks:
            missing_str = ", ".join(missing_blocks)
            print(f"CRITICAL WARNING: Missing prompt blocks: {missing_str}. The generated prompt might be incorrect!")
            raise ValueError(f"Missing required prompt blocks: {missing_str}")

        final_prompt = "\n\n".join(filter(None, parts))
        return final_prompt
