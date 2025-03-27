import chess
from fastapi import FastAPI
from typing import List, Tuple
from board2vec import board2vec, pgn_to_boards

# Создаем экземпляр FastAPI
app = FastAPI()


def moves_to_boards(moves: List[str]):
    boards = []
    board = chess.Board()
    for move in moves:
        board.push(chess.Move.from_uci(move))
        boards.append(board.copy())
    return boards


def model(boards):
    num_pairs = random.randint(1, 5)
    # Создаем список пар случайных чисел
    random_pairs = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_pairs)]
    return random_pairs


# Определяем endpoint
@app.post("/process_moves/")
async def process_moves(moves: str) -> List[Tuple[int, int]]:
    """
    Принимает строку шахматных ходов и возвращает список пар случайных чисел.
    """
    # Вызываем функцию-затычку для генерации пар чисел
    boards = moves_to_boards(moves.split())
    result = model(boards)
    return result

# Для запуска сервиса используйте команду:
# uvicorn <имя_файла>:app --reload