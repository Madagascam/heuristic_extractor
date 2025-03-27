"""
Модуль обрабатывает размеченные шахматные партии, доигрывая их до конца с использованием движка Maia,
если партия не завершилась матом.
"""

import logging
from pathlib import Path
import chess
import chess.engine
import csv
from typing import Dict, Iterator, Union, Literal

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "PATH_TO_LC0_ENGINE": "C:/Users/matvey/Documents/chess_data/lc0/lc0.exe",
    "PATH_TO_MAIA_WEIGHTS_FOLDER": "C:/Users/matvey/workspace/heuristic_extractor/automatic_markup/maia_weights",
}
ENGINE_OPTIONS = {
    "Threads": 1,
}

class MaiaEngine:
    """
    Класс для работы с шахматным движком Maia, основанным на lc0.
    Используется для доигрывания незавершенных партий.

    Пример использования:
        with MaiaEngine() as engine:
            engine.set_output_file('output.csv')
            for game in engine.upload_games('input.csv'):
                updated_game = engine.finish_game(game)
                engine.save(updated_game)
    """
    ratings = [1100, 1300, 1500, 1700, 1900]

    def __init__(self, config: Dict = DEFAULT_CONFIG):
        """
        Инициализирует движок Maia с указанной конфигурацией.

        Args:
            config (Dict): Словарь с настройками (пути к lc0 и весам, количество процессов).
        """
        self.config = config
        self.engines: Dict[int, chess.engine.SimpleEngine] = {}
        self._init_engines()

    def _init_engines(self) -> None:
        """Инициализирует движки lc0 для каждого рейтинга."""
        for rating in self.ratings:
            weights_path = Path(DEFAULT_CONFIG["PATH_TO_MAIA_WEIGHTS_FOLDER"]) / f'maia-{rating}.pb.gz'
            self.engines[rating] = chess.engine.SimpleEngine.popen_uci(
                [DEFAULT_CONFIG['PATH_TO_LC0_ENGINE'], f"--weights={weights_path}"]
            )
            self.engines[rating].configure(ENGINE_OPTIONS)

    def upload_games(self, path_to_markup_csv: str, skip: Union[int, Literal['auto']] = 0) -> Iterator[Dict[str, str]]:
        """
        Загружает партии из CSV-файла.

        Args:
            path_to_markup_csv (str): Путь к CSV-файлу с разметкой партий.

        Yields:
            Dict[str, str]: Словарь с данными одной партии.

        Raises:
            FileNotFoundError: Если файл не найден.
            csv.Error: Если файл поврежден или имеет неверный формат.
        """
        try:
            if skip == 'auto':
                if not hasattr(self, 'file'):
                    raise ValueError('Чтобы использовать skip в автоматическом режиме, сначала используйте метод set_output_file')
                output_path = Path(self.output_path)
                if output_path.exists():
                    with open(self.output_path, 'r') as f:
                        lines = f.readlines()
                        existing_rows = len(lines) - 1  # Вычитаем заголовок
                        skip = max(0, existing_rows)
                else:
                    skip = 0
                logging.info(f'Установлено значение skip={skip}')
            with open(path_to_markup_csv, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self._fieldnames = reader.fieldnames
                for row in reader:
                    if skip > 0:
                        skip -= 1
                        continue
                    yield row
        except Exception as e:
            logging.error(f"Ошибка при чтении CSV-файла {path_to_markup_csv}: {e}")
            raise

    def finish_game(self, row: Dict[str, str]) -> Dict[str, str]:
        """
        Завершает партию до конца, если она не закончилась.

        Args:
            row (Dict[str, str]): Словарь с данными партии, включая 'white_elo', 'black_elo' и 'moves'.

        Returns:
            Dict[str, str]: Обновленный словарь с завершенной последовательностью ходов.

        Raises:
            ValueError: Если рейтинги или ходы имеют неверный формат.
            chess.IllegalMoveError: Если ход невозможен на текущей доске.
        """
        try:
            white_elo = int(row['white_elo']) if row['white_elo'].isdigit() else None
            black_elo = int(row['black_elo']) if row['black_elo'].isdigit() else None
            if white_elo is None or black_elo is None:
                average_elo = 1500
            else:
                average_elo = (white_elo + black_elo) / 2
            moves = row['moves'].split()

            board = chess.Board()
            for move in moves:
                board.push(chess.Move.from_uci(move))

            while not board.is_game_over():
                best_move = self._get_best_move(board, average_elo)
                moves.append(best_move)
                board.push(chess.Move.from_uci(best_move))

            row['moves'] = ' '.join(moves)
            return row
        except Exception as e:
            logging.error(f"Ошибка при завершении партии: {e}")
            raise

    def _get_best_move(self, board: chess.Board, rating: float) -> str:
        """
        Получает лучший ход для текущей позиции от движка.

        Args:
            board (chess.Board): Текущая шахматная позиция.
            rating (float): Средний рейтинг игроков для выбора движка.

        Returns:
            str: Лучший ход в формате UCI.

        Raises:
            chess.engine.EngineError: Если движок не отвечает или завершает работу.
        """
        closest_rating = min(self.ratings, key=lambda x: abs(x - rating))
        engine = self.engines[closest_rating]
        try:
            result = engine.play(board, chess.engine.Limit(time=0.5, nodes=1))
            return result.move.uci()
        except Exception as e:
            logging.error(f"Ошибка при получении хода для рейтинга {closest_rating}: {e}")
            raise

    def set_output_file(self, output_path: str) -> None:
        """
        Устанавливает файл для записи результатов.

        Args:
            output_path (str): Путь к выходному CSV-файлу.
        """
        self.file = open(output_path, mode='a+', encoding='utf-8', newline='')
        self.output_path = output_path
        self.writer = None

    def save(self, row: Dict[str, str]) -> None:
        """
        Сохраняет обновленную партию в выходной файл.

        Args:
            row (Dict[str, str]): Словарь с данными партии для сохранения.
        """
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=self._fieldnames)
            # Проверяем, пуст ли файл
            self.file.seek(0)
            first_line = self.file.readline()
            if not first_line:  # Файл пуст, записываем заголовки
                self.writer.writeheader()
            else:
                # Проверяем совпадение заголовков
                existing_header = first_line.strip().split(',')
                if set(existing_header) != set(self._fieldnames):
                    logging.warning("Структура выходного файла не совпадает с ожидаемой. Заголовки не будут перезаписаны.")
            # Возвращаем указатель в конец файла
            self.file.seek(0, 2)
        self.writer.writerow(row)

    def __enter__(self):
        """Инициализация для использования в with-блоке."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Закрытие всех ресурсов при выходе из with-блока."""
        for engine in self.engines.values():
            engine.quit()
        if hasattr(self, 'file'):
            self.file.close()

# Пример использования с параллельной обработкой
if __name__ == "__main__":
    input_path = 'C:/Users/matvey/workspace/heuristic_extractor/labeled.csv'
    output_path = 'C:/Users/matvey/workspace/heuristic_extractor/labeled2.csv'

    with MaiaEngine() as eng:
        eng.set_output_file(output_path)
        for i, row in enumerate(eng.upload_games(input_path, skip='auto')):
            updated_row = eng.finish_game(row)
            eng.save(updated_row)
            print(f'Обработано {i + 1:>7}...', end='\r')
