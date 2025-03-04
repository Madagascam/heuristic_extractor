import chess
import torch
import numpy as np
from typing import Literal, Union


class BoardEncoder:
    def __init__(self):
        pass

    def encode(
            self,
            board: chess.Board,
            output_type: Literal['torch', 'numpy', 'list']
        ) -> Union[torch.Tensor, np.ndarray, list]:
        pass


class SimpleEncoder(BoardEncoder):
    def __init__(self):
        super(SimpleEncoder, self).__init__()

    def encode(
            self,
            board: chess.Board,
            output_type: Literal['torch', 'numpy', 'list']
        ) -> Union[torch.Tensor, np.ndarray, list]:
        # 1. Кодируем состояние доски (64 значения)
        board_state = np.zeros(64, dtype=np.int8)  # Создаем массив для доски
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece.piece_type
                if piece.color == chess.BLACK:
                    value = -value
                board_state[square] = value
        
        # 2. Добавляем права на рокировку (4 бита)
        castling_rights = np.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ], dtype=np.int8)
        
        # 3. Добавляем возможность взятия на проходе (1 значение)
        en_passant_square = board.ep_square if board.ep_square is not None else -1
        en_passant = np.array([en_passant_square], dtype=np.int8)
        
        # 4. Добавляем текущего игрока (1 бит)
        current_player = np.array([int(board.turn)], dtype=np.int8)  # 1 для белых, 0 для черных
        
        # Объединяем все части в один массив numpy
        vector = np.concatenate([board_state, castling_rights, en_passant, current_player])
        
        # Возвращаем результат в зависимости от запрошенного типа
        if output_type == 'torch':
            return torch.from_numpy(vector)
        elif output_type == 'numpy':
            return vector
        elif output_type == 'list':
            return vector.tolist()
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
        

class SparseEncoder(BoardEncoder):
    def __init__(self):
        super(SparseEncoder, self).__init__()

    def encode(
            self,
            board: chess.Board,
            output_type: Literal['torch', 'numpy', 'list']
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
        
        # Возвращаем результат в зависимости от запрошенного типа
        if output_type == 'torch':
            return torch.tensor(result)
        elif output_type == 'numpy':
            return result
        elif output_type == 'list':
            return result.tolist()
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
    