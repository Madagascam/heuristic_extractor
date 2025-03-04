import torch
import torch.nn as nn
import chess
import numpy as np


class Board2VecInference(nn.Module):
    def __init__(self, weights_path: str, input_dim: int = 774, output_dim: int = 64):
        super(Board2VecInference, self).__init__()

        # Определяем ту же архитектуру для эмбеддингов
        input_hidden_dim = 256
        hidden_dim = 128
        layers = [
            nn.Linear(input_dim, input_hidden_dim),
            nn.ReLU(),
            nn.Linear(input_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ]
        self.embedding = nn.Sequential(*layers)

        self.load_state_dict(torch.load(weights_path, weights_only=False))
        self.eval()
    
    def encode_board(self, board: chess.Board):
        """
        Кодирует шахматную позицию в вектор с битовых карт и дополнительных фич.
        
        Аргументы:
            board (chess.Board): Шахматная позиция.
        Возвращает:
            np.ndarray: Вектор представления позиции длины 774.
        """
        # Инициализируем пустой вектор
        vector = []
        
        # 1. Кодируем состояние доски
        # Битовые карты для фигур (12 карт: 6 типов фигур × 2 цвета)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        colors = [chess.WHITE, chess.BLACK]
        bitboards = []
        for color in colors:
            for piece_type in piece_types:
                bitboard = np.zeros(64, dtype=np.float32)
                for square in board.pieces(piece_type, color):
                    bitboard[square] = 1.0
                bitboards.append(bitboard)
        
        # 2. Добавляем права на рокировку (4 бита)
        castling_rights = [
            board.has_kingside_castling_rights(chess.WHITE),  # Короткая рокировка белых
            board.has_queenside_castling_rights(chess.WHITE), # Длинная рокировка белых
            board.has_kingside_castling_rights(chess.BLACK),  # Короткая рокировка черных
            board.has_queenside_castling_rights(chess.BLACK)  # Длинная рокировка черных
        ]
        vector.extend([int(right) for right in castling_rights])
        
        # 3. Добавляем возможность взятия на проходе (1 значение)
        en_passant_square = board.ep_square  # Индекс клетки для взятия на проходе
        if en_passant_square is not None:
            vector.append(en_passant_square)
        else:
            vector.append(-1)  # Если взятие на проходе невозможно
        
        # 4. Добавляем текущего игрока (1 бит)
        current_player = int(board.turn)  # 1 для белых, 0 для черных
        vector.append(current_player)
        
        return np.concatenate(bitboards + [np.array(vector, dtype=np.float32)])

    def forward(self, board: chess.Board):
        raw_x = self.encode_board(board)
        x = torch.tensor(raw_x)

        # Вычисляем эмбеддинг для входных данных
        return self.embedding(x)


if __name__ == '__main__':
    model = Board2VecInference('./board2vec/board2vec_MLP_weights.pth')
    print(model(chess.Board()))