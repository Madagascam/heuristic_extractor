import chess
import torch
import numpy as np
from typing import Literal, Union, List, Tuple


class BoardEncoder:
    def __init__(self):
        pass

    def encode(
            self,
            board: chess.Board,
            output_type: Literal['torch', 'numpy', 'list']
        ):
        pass

    def make_batch(self, encoded_boards: list):
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
        board_state = np.zeros((8, 8), dtype=np.float32)  # Создаем массив для доски
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece.piece_type
                if piece.color == chess.BLACK:
                    value = -value
                board_state[square // 8][square % 8] = value

        # 2. Добавляем права на рокировку (4 бита)
        castling_rights = np.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ])

        # 3. Добавляем возможность взятия на проходе (1 значение)
        en_passant_square = board.ep_square if board.ep_square is not None else -1
        en_passant = np.array([en_passant_square])

        # 4. Добавляем текущего игрока (1 бит)
        current_player = np.array([float(board.turn)])  # 1 для белых, 0 для черных

        # Объединяем все части в один массив numpy
        adv_vector = np.concatenate([castling_rights, en_passant, current_player], dtype=np.float32)

        # Возвращаем результат в зависимости от запрошенного типа
        if output_type == 'torch':
            return torch.tensor(board_state), torch.tensor(adv_vector)
        elif output_type == 'numpy':
            return board_state, adv_vector
        elif output_type == 'list':
            return board_state.tolist(), adv_vector.tolist()
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

    def make_batch(self, encoded_boards: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        boards, advs = zip(*encoded_boards)
        return torch.tensor(np.array(boards)), torch.tensor(np.array(advs))


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

    def make_batch(self, encoded_boards):
        return torch.tensor(np.array(encoded_boards))
    

class MatrixEncoder:
    def encode(
            self,
            board: chess.Board,
            output_type: Literal['torch', 'numpy', 'list']
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray], Tuple[list, list]]:
        
        # 1. Кодируем состояние доски (12 каналов)
        board_state = np.zeros((12, 8, 8), dtype=np.float32)  # 12 каналов: 6 фигур × 2 цвета
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Определяем канал:
                # 0-5: белые пешка, конь, слон, ладья, ферзь, король
                # 6-11: черные пешка, конь, слон, ладья, ферзь, король
                channel = piece.piece_type - 1
                if piece.color == chess.BLACK:
                    channel += 6
                row = square // 8
                col = square % 8
                board_state[channel, row, col] = 1.0

        # 2. Дополнительные признаки (аналогично предыдущей версии)
        castling_rights = np.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ], dtype=np.float32)

        en_passant = np.array([
            board.ep_square if board.ep_square is not None else -1
        ], dtype=np.float32)

        current_player = np.array([float(board.turn)], dtype=np.float32)  # 1.0 для белых, 0.0 для черных

        adv_vector = np.concatenate([castling_rights, en_passant, current_player])

        # Преобразуем в нужный формат
        if output_type == 'torch':
            return (
                torch.tensor(board_state, dtype=torch.float32),
                torch.tensor(adv_vector, dtype=torch.float32)
            )
        elif output_type == 'numpy':
            return board_state, adv_vector
        elif output_type == 'list':
            return board_state.tolist(), adv_vector.tolist()
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

    def make_batch(self, encoded_boards: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        boards, advs = zip(*encoded_boards)
        if len(boards) > 0 and not isinstance(boards[0], torch.Tensor):
            boards = torch.tensor(boards)
        if len(advs) > 0 and not isinstance(advs[0], torch.Tensor):
            advs = torch.tensor(advs)
        return (
            torch.stack(boards),
            torch.stack(advs)
        )