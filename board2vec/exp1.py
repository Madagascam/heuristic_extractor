import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import chess
import random
import numpy as np
from typing import List
from collections import deque

data = pd.read_csv('./labeled.csv', header=0)
games_series = data['moves'].str.split(' ')

DTYPE = torch.float32


class Board2Vec(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Board2Vec, self).__init__()

        # Эмбеддинги
        input_hidden_dim = 256
        hidden_dim = 128
        layers = [
            nn.Linear(input_dim, input_hidden_dim, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(input_hidden_dim, hidden_dim, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=DTYPE)
        ]
        self.embedding = nn.Sequential(*layers)

    def forward(self, target, context, negatives):
        # Получаем эмбеддинги
        target_embed = self.embedding(target)
        context_embed = self.embedding(context)
        negatives_embed = self.embedding(negatives)

        # Положительные примеры: скалярное произведение между target и context
        pos_scores = torch.mul(target_embed, context_embed).sum(dim=1)
        pos_loss = -torch.nn.functional.logsigmoid(pos_scores)

        # Негативные примеры: скалярное произведение между target и negatives
        neg_scores = torch.bmm(negatives_embed, target_embed.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_scores).sum(dim=1)

        # Общая потеря: усредняем по батчу
        loss = (pos_loss + neg_loss).mean()
        return loss


def encode_board(board: chess.Board):
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


class TargetContextBoardsLoader:
    def __init__(self, games: List[str], window_size: int, game_count: int, pair_cnt: int, subset_size: int):
        self.games = games
        self.WINDOW_SIZE = window_size
        self.PAIR_CNT = pair_cnt
        self.GAME_COUNT = game_count
        self.SUBSET_SIZE = subset_size  # Размер подмножества партий для пула

        self.games_left = self.GAME_COUNT
        self.all_boards = []  # Пул досок из подмножества
        self.boards = []  # Доски текущей игры

        # Инициализируем пул досок в первый раз
        self.update_board_pool()

    def update_board_pool(self):
        """Обновляет пул досок, выбирая случайное подмножество партий."""
        self.all_boards = []
        subset_indices = random.sample(range(len(self.games)), min(self.SUBSET_SIZE, len(self.games)))
        for idx in subset_indices:
            board = chess.Board()
            for move in self.games[idx]:
                board.push(chess.Move.from_uci(move))
                self.all_boards.append(encode_board(board))

    def set_game(self):
        cur_game = random.randint(0, len(self.games) - 1)
        self.games_left -= 1

        self.boards = []
        prev_board = chess.Board()
        game = self.games[cur_game]
        for move in game:
            prev_board.push(chess.Move.from_uci(move))
            encoded = encode_board(prev_board)
            self.boards.append(encoded)
        self.boards = np.array(self.boards, dtype=np.float32)
    
    def gen_pairs(self):
        target = []
        context = []
        negatives = []
        k = 5  # Количество негативных примеров на пару

        for _ in range(self.PAIR_CNT):
            i = random.randint(0, len(self.boards) - 1)
            target.append(self.boards[i])

            # Генерация контекста
            start = max(0, i - self.WINDOW_SIZE)
            end = min(len(self.boards), i + self.WINDOW_SIZE + 1)
            j = random.choice(list(range(start, i)) + list(range(i + 1, end)))
            context.append(self.boards[j])

            # Генерация негативных примеров из пула self.all_boards
            neg_samples = random.sample(self.all_boards, k)
            negatives.append(neg_samples)

        target = np.array(target, dtype=np.float32)
        context = np.array(context, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)

        return (torch.tensor(target, dtype=DTYPE),
                torch.tensor(context, dtype=DTYPE),
                torch.tensor(negatives, dtype=DTYPE))

    def __iter__(self):
        return self

    def __next__(self):
        if self.games_left <= 0:
            self.games_left = self.GAME_COUNT
            raise StopIteration
        
        self.set_game()
        
        return self.gen_pairs()
    

# Модель
input_dim = 774
output_dim = 64
model = Board2Vec(input_dim, output_dim)

# Даталоадер
dataloader = TargetContextBoardsLoader(games_series, window_size=8, game_count=50, pair_cnt=50, subset_size=1000)

# Гиперпараметры
learning_rate = 0.001
num_epochs = 10000

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Планировщик
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95)

# Цикл обучения
for epoch in range(num_epochs):
    total_loss = 0
    for target, context, negatives in dataloader:
        loss = model(target, context, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    
    # Обновляем пул досок после каждой эпохи
    dataloader.update_board_pool()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}, lr: {scheduler.get_last_lr()}")
    if (epoch + 1) % 10 == 0:
        # Сохраняем веса модели
        print(f'Веса модели сохранены в файл "board2vec_weights.pth"')
        torch.save(model.state_dict(), "board2vec_weights.pth")

