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
    winning chances from -1 to 1
    """
    mate = score.mate()
    if mate is not None:
        return 1 if mate > 0 else -1

    cp = score.score()
    MULTIPLIER = -0.00368208
    return 2 / (1 + math.exp(MULTIPLIER * cp)) - 1 if cp is not None else 0


def get_next_move_pair(engine: chess.engine.SimpleEngine, node: chess.pgn.GameNode, winner: chess.Color, limit: chess.engine.Limit) -> NextMovePair:
    info = engine.analyse(node.board(), multipv = 2, limit = limit)
    best = EngineMove(info[0]["pv"][0], info[0]["score"].pov(winner))
    second = EngineMove(info[1]["pv"][0], info[1]["score"].pov(winner)) if len(info) > 1 else None
    return NextMovePair(node, winner, best, second)

def count_mates(board: chess.Board) -> int:
    mates = 0
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mates += 1
        board.pop()
    return mates

def get_tier(game: chess.pgn.Game) -> int:
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
            return 

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