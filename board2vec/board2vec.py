import os
import chess
import chess.pgn
from .experiments.experiment_04.inference import board2vec
from typing import List


def pgn_to_boards(path_to_pgn: str) -> List[List[chess.Board]]:
    boards = []
    if not os.path.exists(path_to_pgn):
        raise ValueError(f'Файла {pgn} не существует')

    with open(path_to_pgn, 'r', encoding='utf-8') as pgn_file:
        while (game := chess.pgn.read_game(pgn_file)):
            board = chess.Board()
            res = []
            for move in game.mainline_moves():
                board.push(move)
                res.append(board.copy())
            boards.append(res)

    return boards


def game_to_boards(game: chess.pgn.Game) -> List[chess.Board]:
    board = chess.Board()
    res = []
    for move in game.mainline_moves():
        board.push(move)
        res.append(board.copy())
    return res