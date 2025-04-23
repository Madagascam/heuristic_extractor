import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional
import time

from core.pgn_parser import PGNParser
from core.response_verifier import ResponseVerifier, VerificationError
from providers.google_provider import GoogleProvider
from core.interfaces import LLMServiceProvider, ModelConfig, PromptParams, RetryConfig, LLMGenerationError

#1. Конфигурация

# Путь к датасету
DATA_FILE = 'full_labeled.csv'

# Google API Key
GOOGLE_API_KEY = ""

# Параметры LLM (Google GenAI)
llm_provider: LLMServiceProvider = GoogleProvider(api_key=GOOGLE_API_KEY)
model_config: ModelConfig = {'model_name': 'gemini-2.0-flash'}
retry_config: RetryConfig = {'max_attempts': 3, 'delay_seconds': 5}

# Параметры Эксперимента
N_FEW_SHOT_EXAMPLES = 29
N_TEST_SAMPLES = 6      # Количество партий для оценки каждой комбинации промпта/примеров
RANDOM_SEED = 42

# Параметры Стратификации (из EDA)
N_ELO_BINS = 3
N_LEN_BINS = 3
N_START_BINS = 3

llm_prompt_params: PromptParams = {
    'scope': 'single',
    'focus': 'generic_highlight',
    'output_format': 'simple',
}

parser = PGNParser()
verifier = ResponseVerifier()

#2. Загрузка и Подготовка Данных

def load_and_prepare_data(filepath: str) -> Optional[pd.DataFrame]:
    """Загружает CSV, выполняет базовую очистку и инженерию признаков."""
    try:
        df = pd.read_csv(filepath)
        print(f"Загружено {df.shape[0]} строк из {filepath}")

        df.dropna(subset=['white_elo', 'black_elo'], inplace=True)

        df['game_len_halfmoves'] = df['moves'].apply(lambda x: len(x.split()) if pd.notna(x) and x else 0)

        def parse_marks_simple(marks_str):
            if pd.isna(marks_str): return None, None
            try:
                start, end = map(int, marks_str.split(','))
                return start, end
            except: return None, None

        marks_parsed = df['marks'].apply(parse_marks_simple)
        df['highlight_start'] = marks_parsed.apply(lambda x: x[0])
        df['highlight_end'] = marks_parsed.apply(lambda x: x[1])
        df.dropna(subset=['highlight_start', 'highlight_end'], inplace=True)
        df['highlight_start'] = df['highlight_start'].astype(int)
        df['highlight_end'] = df['highlight_end'].astype(int)

        # Добавляем нужные колонки для стратификации
        df['avg_elo'] = (df['white_elo'] + df['black_elo']) / 2
        df['highlight_start_percent'] = (df['highlight_start'] / df['game_len_halfmoves']).fillna(0).clip(0, 1)

        print(f"Осталось {df.shape[0]} строк после очистки и подготовки.")
        return df

    except FileNotFoundError:
        print(f"Ошибка: Файл '{filepath}' не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке или подготовке данных: {e}")
        return None

#3. Функции Выборки ID

def select_stratified_ids(df: pd.DataFrame, n_samples: int, seed: int) -> List[str]:
    """Выполняет стратифицированную выборку ID."""
    print(f"Выполняется стратифицированная выборка {n_samples} ID...")
    if df is None or df.empty: return []

    n_strata = N_ELO_BINS * N_LEN_BINS * N_START_BINS
    n_samples_per_stratum = int(np.ceil(n_samples / n_strata))
    selected_ids = []

    try:
        df_strat = df.copy()
        df_strat['elo_stratum'] = pd.qcut(df_strat['avg_elo'], q=N_ELO_BINS, labels=False, duplicates='drop')
        df_strat['len_stratum'] = pd.qcut(df_strat['game_len_halfmoves'], q=N_LEN_BINS, labels=False, duplicates='drop')
        df_strat['start_stratum'] = pd.qcut(df_strat['highlight_start_percent'], q=N_START_BINS, labels=False, duplicates='drop')

        grouped = df_strat.groupby(['elo_stratum', 'len_stratum', 'start_stratum'], observed=False)
        selected_samples_list = []
        for _, group in grouped:
            n_take = min(n_samples_per_stratum, len(group))
            if n_take > 0:
                selected_samples_list.append(group.sample(n=n_take, random_state=seed))

        if selected_samples_list:
            selected_df = pd.concat(selected_samples_list).reset_index(drop=True)
            selected_df = selected_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            selected_ids = selected_df.head(n_samples)['id'].tolist()
            print(f"Выбрано {len(selected_ids)} стратифицированных ID.")
        else:
            print("Не удалось выбрать стратифицированные примеры.")

    except Exception as e:
        print(f"Ошибка при стратифицированной выборке: {e}. Возвращаем пустой список.")
        selected_ids = []

    return selected_ids


def select_random_ids(df: pd.DataFrame, n_samples: int, exclude_ids: List[str], seed: int) -> List[str]:
    """Выбирает случайные ID, исключая заданные."""
    print(f"Выполняется случайная выборка {n_samples} ID...")
    if df is None or df.empty: return []

    available_ids = df[~df['id'].isin(exclude_ids)]['id'].unique()
    if len(available_ids) < n_samples:
        print(f"Предупреждение: Недостаточно уникальных ID ({len(available_ids)}) для случайной выборки {n_samples}.")
        n_samples = len(available_ids)

    random.seed(seed)
    selected_ids = random.sample(list(available_ids), n_samples)
    print(f"Выбрано {len(selected_ids)} случайных ID.")
    return selected_ids

#4. Подготовка Данных для Промпта и Оценки

def format_example_pgn_style(row: pd.Series) -> str:
    """Форматирует пример в стиле PGN для few-shot."""
    try:
        moves_str = parser.parse(row['moves'])
        target_output = f"[{row['highlight_start']}, {row['highlight_end']}]"
        return f"Game:\n{moves_str}\nCorrect Highlight Output:\n{target_output}"
    except Exception as e:
        print(f"Ошибка форматирования PGN-стиля для ID {row.get('id', 'N/A')}: {e}")
        return ""

def format_example_csv_style(row: pd.Series) -> str:
    """Форматирует пример в стиле CSV строки для few-shot."""
    target_output = f"[{row['highlight_start']}, {row['highlight_end']}]"
    return f"{row.get('white_elo', 0)},{row.get('black_elo', 0)},{row.get('moves', '')},{target_output}"

def get_input_data_for_llm(row: pd.Series, prompt_style: str) -> str:
    """Возвращает данные для запроса к LLM в нужном формате."""
    if prompt_style == 'pgn':
        try:
            return parser.parse(row['moves'])
        except Exception:
            return ""
    elif prompt_style == 'csv':
        return f"{row.get('white_elo', 0)},{row.get('black_elo', 0)},{row.get('moves', '')}"
    else:
        return ""

#5. Генерация Промптов

def generate_few_shot_prompt(prompt_params: PromptParams, examples: List[str], input_data: str) -> str:
    """Собирает промпт с few-shot примерами."""
    prompt_lines = [
        "You are a chess analysis assistant.",
        "Your task is to identify the single most interesting segment (max 20 half-moves) in the provided chess game.",
        "Output ONLY a valid JSON list containing exactly two integers: [start_halfmove, end_halfmove]",
        "Do not include any other text, explanations, or markdown formatting.",
        "\nHere are some examples of input and expected output:",
    ]
    # Добавляем примеры
    prompt_lines.extend(examples)

    # Добавляем текущий запрос
    prompt_lines.append("\nNow, analyze the following game:")
    prompt_lines.append(input_data)
    prompt_lines.append("\nJSON Output:")

    return "\n\n".join(prompt_lines)

#6. Основной Цикл Оценки

def evaluate_llm(df: pd.DataFrame, few_shot_ids: List[str], test_ids: List[str], prompt_style: str, examples_style: str) -> List[Dict]:
    """Формирует промпты, получает ответы LLM и верифицирует их для тестового набора."""
    results = []
    print(f"\n--- Оценка для: Prompt Style='{prompt_style}', Examples Style='{examples_style}' ---")

    few_shot_df = df[df['id'].isin(few_shot_ids)]

    examples = []
    if examples_style == 'pgn':
        examples = [format_example_pgn_style(row) for _, row in few_shot_df.iterrows() if format_example_pgn_style(row)]
    elif examples_style == 'csv':
        examples = ['white_elo,black_elo,moves,marks']
        examples.extend([format_example_csv_style(row) for _, row in few_shot_df.iterrows()])
    print(f"Подготовлено {len(examples)} few-shot примеров.")
    if not examples:
         print("Предупреждение: Не удалось создать few-shot примеры.")

    test_df = df[df['id'].isin(test_ids)].copy()
    print(f"Обработка {len(test_df)} тестовых примеров...")

    for index, row in test_df.iterrows():
        game_id = row['id']
        print(f"  Обработка ID: {game_id}...")
        input_data = get_input_data_for_llm(row, prompt_style)
        if not input_data:
             print(f"    Пропуск ID: {game_id} (не удалось получить входные данные)")
             results.append({'id': game_id, 'error': 'Input data generation failed', 'prediction': None, 'target': [row['highlight_start'], row['highlight_end']]})
             continue

        full_prompt = generate_few_shot_prompt(llm_prompt_params, examples, input_data)

        llm_prediction = None
        error_message = None
        try:
            raw_response = llm_provider.generate_response(full_prompt, model_config, retry_config)
            llm_prediction = verifier.verify(raw_response, llm_prompt_params)

        except (LLMGenerationError, VerificationError, Exception) as e:
             print(f"    Ошибка для ID {game_id}: {type(e).__name__}: {e}")
             error_message = f"{type(e).__name__}: {str(e)[:100]}"
        except KeyboardInterrupt:
             print("Прервано пользователем.")
             raise

        results.append({
            'id': game_id,
            'error': error_message,
            'prediction': llm_prediction,
            'target': [row['highlight_start'], row['highlight_end']]
        })
        time.sleep(10)

    return results

#7. Запуск и Сохранение Результатов

if __name__ == "__main__":
    print("Запуск скрипта оценки GPT...")
    main_df = load_and_prepare_data(DATA_FILE)

    if main_df is None:
        print("Не удалось загрузить данные. Завершение скрипта.")
        exit()

    stratified_few_shot_ids = select_stratified_ids(main_df, N_FEW_SHOT_EXAMPLES, seed=RANDOM_SEED)
    random_few_shot_ids = select_random_ids(main_df, N_FEW_SHOT_EXAMPLES, exclude_ids=stratified_few_shot_ids, seed=RANDOM_SEED + 1)

    all_few_shot_ids = stratified_few_shot_ids + random_few_shot_ids
    test_set_ids = select_random_ids(main_df, N_TEST_SAMPLES, exclude_ids=all_few_shot_ids, seed=RANDOM_SEED + 2)

    if not test_set_ids:
        print("Не удалось выбрать тестовые ID. Завершение скрипта.")
        exit()

    all_results = {}

    # Комбинация 1: Стратифицированные примеры, PGN стиль промпта
    results_strat_pgn = evaluate_llm(main_df, stratified_few_shot_ids, test_set_ids, prompt_style='pgn', examples_style='pgn')
    all_results['stratified_pgn'] = results_strat_pgn

    # Комбинация 2: Стратифицированные примеры, CSV стиль промпта
    results_strat_csv = evaluate_llm(main_df, stratified_few_shot_ids, test_set_ids, prompt_style='csv', examples_style='csv')
    all_results['stratified_csv'] = results_strat_csv

    # Комбинация 3: Случайные примеры, PGN стиль промпта
    results_random_pgn = evaluate_llm(main_df, random_few_shot_ids, test_set_ids, prompt_style='pgn', examples_style='pgn')
    all_results['random_pgn'] = results_random_pgn

    # Комбинация 4: Случайные примеры, CSV стиль промпта
    results_random_csv = evaluate_llm(main_df, random_few_shot_ids, test_set_ids, prompt_style='csv', examples_style='csv')
    all_results['random_csv'] = results_random_csv

    #Сохранение Результатов
    results_list_for_df = []
    for config_name, results in all_results.items():
        for res in results:
            res['config'] = config_name
            results_list_for_df.append(res)

    results_df = pd.DataFrame(results_list_for_df)
    output_filename = 'gpt_evaluation_results2.csv'
    try:
        results_df.to_csv(output_filename, index=False)
        print(f"\nРезультаты оценки сохранены в файл: {output_filename}")
        print("\nПример сохраненных результатов:")
        display(results_df.head())
    except Exception as e:
        print(f"\nОшибка при сохранении результатов в CSV: {e}")

    print("\nСкрипт завершен.")