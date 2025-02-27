"""
Модуль обрабатывает размеченные партии, "доигрывая"
их до конца, если друг игра (последовательность ходов)
не заканчивается матом.
"""

import chess
from dataclasses import dataclass
import subprocess
import select
import json
import os
import signal
import time

CONFIG = {
    "PATH_TO_LC0_ENGINE": "lc0/lc0.exe",
    "PATH_TO_MAIA_WEIGHTS_FOLDER": "maia_weights",
    "NUMBER_OF_PROCESSES": 5,
}


class MaiaEngine:
    """
    with MaiaEngine() as engine:
        engine.start_processes()
    """
    index_of_process_to_rating = {1100 + i: i // 200 for i in range(0, 801, 200)}

    def __init__(self):
        self.lc0_engines = []
        self._init_lc0_engines()

        self.board = None
        self.current_lc0_engine_index = 0
        self.game_info = None

    def _init_lc0_engines(self) -> None:
        """
        Инициализирует процессы lc0 для разных рейтингов. Создаёт и запускает процессы,
        загружает веса для каждой модели и ожидает подтверждения "uciok" от каждого процесса.

        Исключения:
            TimeoutError: Если процесс не отправил "uciok" в течение 5 секунд.
            Exception: Если возникла ошибка при запуске процессов.
        """
        try:
            for i in range(CONFIG["NUMBER_OF_PROCESSES"]):
                command = [CONFIG["PATH_TO_LC0_ENGINE"], f"--weights=maia_weights/maia-{1100 + 200 * i}.pb.gz"]
                process = subprocess.Popen(command,
                                           stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           text=True)
                self.lc0_engines.append(process)

                process.stdin.write("uci\n")
                process.stdin.flush()

                # Чтение и ожидание "uciok"
                start_time = time.time()
                timeout = 5  # Время ожидания в секундах

                while True:
                    # Проверяем, не завершился ли процесс
                    if process.poll() is not None:
                        print("Процесс завершён неожиданно.")
                        raise TimeoutError

                    output = process.stdout.readline().strip()
                    if output == "uciok":
                        break

                    # Проверка таймаута
                    if time.time() - start_time > timeout:
                        print("Ошибка: не получили 'uciok' за 5 секунд.")
                        raise TimeoutError

        except Exception as e:
            self._close_all_processes()
            raise e

    def upload_game(self, path_to_markup_json) -> None:
        """
        Загружает партию из указанного JSON-файла, обновляет шахматную доску и выбирает
        движок lc0 в зависимости от среднего рейтинга игроков.

        Параметры:
            path_to_markup_json (str): Путь к файлу JSON с разметкой партии.

        Исключения:
            Exception: Если возникает ошибка при чтении файла или обработке данных.
        """
        try:
            with open(path_to_markup_json, mode='r', encoding='UTF-8') as file:
                data = json.load(file)

            self.game_info = data
            self.board = chess.Board()
            # Проигрываем все ходы
            for move in data["moves"]:
                self.board.push(chess.Move.from_uci(move))

            ratings = list(self.index_of_process_to_rating.keys())
            average_elo = round((data['white_elo'] + data['black_elo']) / 2)
            self.current_lc0_engine_index = self.index_of_process_to_rating[
                min(ratings, key=lambda x: abs(x - average_elo))]

        except Exception as e:
            print(e)

    def _close_all_processes(self) -> None:
        """
        Безопасно завершает все процессы lc0, отправляя команду "quit" и ожидая их завершения.

        Исключения:
            Exception: Если процесс не завершился должным образом, может быть принудительно завершён.
        """
        for process in self.lc0_engines:
            if process.poll() is None:  # Процесс ещё выполняется
                try:
                    # Сначала отправляем quit в stdin
                    if process.stdin:
                        process.stdin.write("quit\n")
                        process.stdin.flush()

                    # Даем время для завершения
                    process.wait(timeout=1)

                except Exception:
                    # Принудительно завершаем процесс, если он не завершился
                    if os.name == "nt":
                        process.terminate()  # Windows
                    else:
                        os.kill(process.pid, signal.SIGTERM)  # Linux/Mac

                # Гарантия, что процесс точно закрыт
                process.kill()
                process.wait()
        self.lc0_engines = []

    def finish_game(self) -> None:
        """
        Завершающий метод игры. Делает ходы до тех пор, пока игра не закончится.

        Исключения:
            ValueError: Если шахматная доска не была инициализирована перед завершением игры.
        """
        if self.board is None:
            self._close_all_processes()
            raise ValueError

        while not self.board.is_game_over():
            self._make_best_move()

    def _make_best_move(self) -> None:
        """
        Делает лучший ход для текущей позиции, используя движок lc0. Ожидает лучший ход
        от движка и применяет его к доске.

        Исключения:
            TimeoutError: Если движок не успевает предоставить лучший ход в течение 5 секунд.
        """
        if self.board.is_game_over():
            return

        fen = self.board.fen()
        engine = self.lc0_engines[self.current_lc0_engine_index]

        engine.stdin.write(f"position fen {fen}\n")  # Устанавливаем позицию
        engine.stdin.write("go nodes 1\n")  # Предсказание следующего хода
        engine.stdin.flush()  # Принудительная отправка данных
        timeout = 2
        best_move = ''

        # Чтение и ожидание "uciok"
        start_time = time.time()
        timeout = 5  # Время ожидания в секундах

        while True:
            line = engine.stdout.readline().strip()
            if "bestmove" in line:  # Останавливаемся, когда появляется лучший ход
                best_move = line.split()[1]
                break

            # Проверка таймаута
            if time.time() - start_time > timeout:
                self._close_all_processes()
                print("TimeOut")
                raise TimeoutError
        # Не получили ход
        if best_move == '':
            self._close_all_processes()
            raise TimeoutError

        self.board.push(chess.Move.from_uci(best_move))
        # Добавляем ход в размеченные данные
        self.game_info['moves'].append(best_move)

    def output_info(self, path_to_output):
        with open(path_to_output, mode='w', encoding='UTF-8') as file:
            json.dump(self.game_info, file, indent=4)

    def __enter__(self):
        """Метод для использования в with-блоке. Инициализация объекта."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Метод для использования в with-блоке. Завершение работы объекта."""
        self._close_all_processes()

    def __del__(self):
        """Деструктор: вызывается при удалении объекта."""
        self._close_all_processes()


# пример использования
if __name__ == "__main__":
    with MaiaEngine() as eng:
        eng.upload_game(f"example_markup1.json")
        eng.finish_game()
        eng.output_info("output.json")

        eng.upload_game(f"example_markup2.json")
        eng.finish_game()
        eng.output_info("output.json")
