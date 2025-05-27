import chess  # Для работы с шахматной доской и правилами
import numpy as np  # Для работы с многомерными массивами
from typing import Literal, Union, List, Tuple  # Для аннотаций типов


class MatrixEncoder:
    """
    Класс для преобразования шахматной доски в матричное представление.
    Каждая фигура на доске кодируется в отдельном канале матрицы 8x8.
    """
    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Кодирует текущее состояние шахматной доски в матрицу 12x8x8.
        Где 12 каналов соответствуют 6 типам фигур для каждого цвета.
        
        Args:
            board: Шахматная доска (объект chess.Board)
            
        Returns:
            Трехмерный массив numpy (12, 8, 8), где:
            - 12 каналов (6 типов фигур × 2 цвета)
            - 8x8 - шахматная доска
        """
        # 12 каналов для фигур (6 типов × 2 цвета)
        board_state = np.zeros((12, 8, 8), dtype=np.float32)

        # Кодируем состояние доски
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Определяем канал:
                # 0-5: белые пешка, конь, слон, ладья, ферзь, король
                # 6-11: черные пешка, конь, слон, ладья, ферзь, король
                channel = piece.piece_type - 1
                if piece.color == chess.BLACK:
                    channel += 6
                # Преобразуем координаты квадрата в строку и столбец
                row = square // 8
                col = square % 8
                board_state[channel, row, col] = 1.0

        return board_state

    def get_encoded_shape(self) -> Tuple[int, int, int]:
        """
        Возвращает форму закодированной матрицы.
        
        Returns:
            Кортеж с размерностями (12, 8, 8)
        """
        return (12, 8, 8)