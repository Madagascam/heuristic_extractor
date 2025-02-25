import math
import chess
import chess.pgn
import chess.engine
from dataclasses import dataclass
from typing import Optional

pair_limit = chess.engine.Limit(depth = 50, time = 30, nodes = 25_000_000)
mate_defense_limit = chess.engine.Limit(depth = 15, time = 10, nodes = 8_000_000)

@dataclass
class EngineMove:
    move: chess.Move
    score: chess.engine.Score

@dataclass
class NextMovePair:
    node: chess.pgn.GameNode
    winner: chess.Color
    best: EngineMove
    second: Optional[EngineMove]


def win_chances(score: chess.engine.Score) -> float:
    """
    Рассчитывает вероятность победы на основе оценки позиции.

    Parameters:
        score (chess.engine.Score): Оценка позиции, полученная от шахматного движка.
            Может быть представлена в виде очков центипешек (cp) или глубины матового расчёта (mate).

    Returns:
        float: Вероятность победы в диапазоне от -1 до 1.
            Значение ближе к 1 указывает на высокую вероятность победы текущей стороны,
            значение ближе к -1 указывает на высокую вероятность поражения текущей стороны.

    Notes:
        Если оценка указывает на мат через N ходов, функция возвращает ±1 в зависимости от знака.
        Для оценок в центипешках используется логистическая функция с коэффициентом `-0.00368208`.
    """
    mate = score.mate()
    if mate is not None:
        return 1 if mate > 0 else -1

    cp = score.score()
    MULTIPLIER = -0.00368208
    return 2 / (1 + math.exp(MULTIPLIER * cp)) - 1 if cp is not None else 0


def get_next_move_pair(engine: chess.engine.SimpleEngine, 
                       node: chess.pgn.GameNode, 
                       winner: chess.Color, 
                       limit: chess.engine.Limit) -> NextMovePair:
    """
    Анализирует следующие два лучших хода из текущей позиции и возвращает их вместе с информацией о победителе.

    Parameters:
        engine (chess.engine.SimpleEngine): Шахматный движок, используемый для анализа.
        node (chess.pgn.GameNode): Узел игры, представляющий текущую позицию.
        winner (chess.Color): Цвет игрока, который считается победителем в данной позиции.
        limit (chess.engine.Limit): Лимит времени/глубины для анализа.

    Returns:
        NextMovePair: Объект, содержащий:
            - `node`: Исходный узел игры.
            - `winner`: Цвет победителя.
            - `best`: Лучший ход (первый по качеству).
            - `second`: Второй лучший ход (если доступен).

    Notes:
        Функция использует движок для анализа двух лучших ходов (`multipv=2`) и преобразует результаты в объект `NextMovePair`.
    """
    info = engine.analyse(node.board(), multipv = 2, limit = limit)
    best = EngineMove(info[0]["pv"][0], info[0]["score"].pov(winner))
    second = EngineMove(info[1]["pv"][0], info[1]["score"].pov(winner)) if len(info) > 1 else None
    return NextMovePair(node, winner, best, second)

def count_mates(board: chess.Board) -> int:
    """
    Подсчитывает количество возможных матовых позиций после каждого легального хода.

    Parameters:
        board (chess.Board): Текущая шахматная доска.

    Returns:
        int: Количество матовых позиций, которые могут быть достигнуты после одного хода.

    Notes:
        Функция перебирает все легальные ходы на доске, применяет каждый ход и проверяет,
        приводит ли он к мату. После анализа ход отменяется, чтобы сохранить исходное состояние доски.
    """
    mates = 0
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mates += 1
        board.pop()
    return mates

def get_tier(game: chess.pgn.Game) -> int:
    """
    Определяет тир (уровень) игры на основе времени контроля и рейтингов игроков.

    Parameters:
        game (chess.pgn.Game): Игра, представленная в формате PGN.

    Returns:
        int: Числовое значение тира игры в диапазоне от 0 до 3:
            - 0: Низкий тир (быстрые партии или низкие рейтинги).
            - 1: Средний тир.
            - 2: Выше среднего тир.
            - 3: Высокий тир.

    Notes:
        Тир определяется как минимальное значение между временем контроля и рейтингами игроков:
        - Время контроля переводится в секунды, учитывая основное время и инкремент.
        - Рейтинги игроков классифицируются в диапазоны (например, >1750 — высокий рейтинг).
        Если данные отсутствуют или некорректны, функция возвращает тир 0.
    """
    # Извлечение заголовков из игры
    headers = game.headers

    # Определение тира по времени контроля
    time_control = headers.get("TimeControl", "0+0")
    try:
        seconds, increment = time_control.split("+")
        total_time = int(seconds) + int(increment) * 40  # Предполагаем 40 ходов
        if total_time >= 480:
            time_tier = 3
        elif total_time >= 180:
            time_tier = 2
        elif total_time > 60:
            time_tier = 1
        else:
            time_tier = 0
    except Exception:
        time_tier = 0

    # Определение тира по рейтингам
    def get_rating_tier(rating: str) -> int:
        try:
            rating_value = int(rating)
        except ValueError:
            return 0

        if rating_value > 1750:
            return 3
        if rating_value > 1600:
            return 2
        if rating_value > 1500:
            return 1

        return 0

    white_rating = headers.get("WhiteElo", "0")
    black_rating = headers.get("BlackElo", "0")
    white_tier = get_rating_tier(white_rating)
    black_tier = get_rating_tier(black_rating)

    # Берём минимальный тир между белым и чёрным
    rating_tier_value = min(white_tier, black_tier)

    # Общий тир — это минимальное значение между временем и рейтингами
    return min(time_tier, rating_tier_value)