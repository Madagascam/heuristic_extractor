import chess
import torch
import numpy as np
from typing import Literal, Union, List, Tuple


class BoardEncoder:
    def __init__(self):
        pass

    def encode(self, board: chess.Board):
        pass

    def get_encoded_shape(self):
        pass


class SparseEncoder(BoardEncoder):
    def __init__(self):
        super(SparseEncoder, self).__init__()

    def encode(
            self,
            board: chess.Board
        ) -> Union[torch.Tensor, np.ndarray, list]:
        # 1. Кодируем состояние доски через битовые карты (12 карт: 6 типов фигур × 2 цвета)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        colors = [chess.WHITE, chess.BLACK]

        # Создаем массив для всех битовых карт (12 × 64)
        bitboards = np.zeros((len(colors) * len(piece_types), 64), dtype=np.float32)

        for color_idx, color in enumerate(colors):
            for piece_idx, piece_type in enumerate(piece_types):
                index = color_idx * len(piece_types) + piece_idx
                for square in board.pieces(piece_type, color):
                    bitboards[index, square] = 1.0

        # 2. Добавляем права на рокировку (4 бита)
        castling_rights = np.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ], dtype=np.float32)

        # 3. Добавляем возможность взятия на проходе (1 значение)
        en_passant_square = np.array([board.ep_square if board.ep_square is not None else -1], dtype=np.float32)

        # 4. Добавляем текущего игрока (1 бит)
        current_player = np.array([int(board.turn)], dtype=np.float32)

        # Объединяем все данные в один массив
        result = np.concatenate([bitboards.flatten(), castling_rights, en_passant_square, current_player])

        return result

    def get_encoded_shape(self):
        return (774,)


class MatrixEncoder(BoardEncoder):
    def encode(self, board: chess.Board) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[np.ndarray, np.ndarray],
        Tuple[list, list]
    ]:
        # 9 каналов: 6 фигур + право рокировки + взятие на проходе + ход пешки на две клетки вперед
        board_state = np.zeros((9, 8, 8), dtype=np.float32)

        # 1. Кодируем состояние доски (6 каналов)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Определяем канал:
                # 0-5: пешка, конь, слон, ладья, ферзь, король
                channel = piece.piece_type - 1
                row = square // 8
                col = square % 8
                board_state[channel, row, col] = 1.0 if piece.color == chess.WHITE else -1.0

        # 2. Дополнительные признаки
        if board.has_kingside_castling_rights(chess.WHITE):
            board_state[6][7, 4] = 1.0  # Король белых на e1
        if board.has_queenside_castling_rights(chess.WHITE):
            board_state[6][7, 4] = 1.0  # Король белых на e1
        if board.has_kingside_castling_rights(chess.BLACK):
            board_state[6][7, 0] = -1.0  # Король чёрных на e8
        if board.has_queenside_castling_rights(chess.BLACK):
            board_state[6][7, 0] = -1.0  # Король чёрных на e8

        if board.ep_square is not None:
            ep_row = board.ep_square // 8
            ep_col = board.ep_square % 8
            board_state[7][ep_row, ep_col] = 1.0

        if board.peek() and board.peek().promotion is None:
            last_move = board.peek()
            if abs(last_move.from_square - last_move.to_square) == 16:  # Ход на две клетки
                double_move_row = last_move.to_square // 8
                double_move_col = last_move.to_square % 8
                board_state[8][double_move_row, double_move_col] = 1.0

        return board_state

    def get_encoded_shape(self):
        return (9, 8, 8)