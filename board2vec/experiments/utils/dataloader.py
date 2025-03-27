import chess
import chess.engine
import random
import chess.engine
import torch
import numpy as np
import pandas as pd
from typing import List
from .board_encoder import BoardEncoder


class TargetContextBoardsLoader:
    def __init__(
            self, games: List[List[str]],
            board_encoder: BoardEncoder,
            window_size: int,
            game_count: int,
            pair_cnt: int,
            subset_size: int,
            negatives_cnt: int,
            device
    ):
        self.games = [game for game in games if self.check_game(game)]
        self.board_encoder = board_encoder
        self.WINDOW_SIZE = window_size
        self.PAIR_CNT = pair_cnt
        self.GAME_COUNT = game_count
        # Размер подмножества партий для пула
        self.SUBSET_SIZE = subset_size
        self.NEGATIVES_CNT = negatives_cnt

        self.games_left = self.GAME_COUNT
        # Пул досок из подмножества
        self.pool_encoded_boards = np.empty(
            (self.SUBSET_SIZE, *self.board_encoder.get_encoded_shape()),
            dtype=np.float32,
            order='C'
        )
        # Доски текущей игры
        self.encoded_boards = None

        self.device = device

        # Инициализируем пул досок в первый раз
        self.update_board_pool()

    def check_game(self, game):
        flag = True
        for i, move in enumerate(game):
            try:
                chess.Move.from_uci(move)
            except chess.InvalidMoveError:
                if move == '' and i == len(game) - 1:
                    del game[-1]
                    break
                else:
                    flag = False
        return flag

    def update_board_pool(self):
        """Обновляет пул досок, выбирая случайное подмножество партий."""
        for i in range(self.SUBSET_SIZE):
            idx = random.randint(0, len(self.games) - 1)
            board = chess.Board()
            for move in self.games[idx]:
                board.push(chess.Move.from_uci(move))
                self.pool_encoded_boards[i] = self.board_encoder.encode(board)

    def set_game(self):
        cur_game = random.randint(0, len(self.games) - 1)
        self.games_left -= 1
        game = self.games[cur_game]

        self.boards = []
        self.encoded_boards = np.empty(
            (len(game), *self.board_encoder.get_encoded_shape()),
            dtype=np.float32,
            order='C'
        )
        prev_board = chess.Board()
        for i, move in enumerate(game):
            prev_board.push(chess.Move.from_uci(move))
            self.boards.append(prev_board.copy())
            self.encoded_boards[i] = self.board_encoder.encode(prev_board)

    def gen_pairs(self):
        # индексы для таргета
        i_indices = np.random.randint(0, len(self.encoded_boards), size=self.PAIR_CNT)

        # индексы для контекста
        start = np.maximum(0, i_indices - self.WINDOW_SIZE)
        end = np.minimum(len(self.encoded_boards) - 1, i_indices + self.WINDOW_SIZE)
        j_indices = np.random.randint(start, end, size=self.PAIR_CNT)
        j_indices[j_indices >= i_indices] += 1

        # индексы для негативных примеров
        neg_indices = np.random.choice(
            len(self.pool_encoded_boards), 
            size=(self.PAIR_CNT, self.NEGATIVES_CNT)
        )

        target = self.encoded_boards[i_indices]
        context = self.encoded_boards[j_indices]
        negatives = self.pool_encoded_boards[neg_indices]

        return (
            torch.tensor(target, device=self.device),
            torch.tensor(context, device=self.device),
            torch.tensor(negatives, device=self.device)
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.games_left <= 0:
            self.games_left = self.GAME_COUNT
            self.update_board_pool()
            raise StopIteration

        self.set_game()

        return self.gen_pairs()


"""
НУЖНО ПЕРЕПИСАТЬ, рабочий вариант кода в коммите 174c3b31694f4dfa8863c41a6e2c781ebbca3af4
(
    Ответственный: Матвей
    Телеграмм: @kend27
    Почта: matvey.ivchenko@yandex.ru
)
"""
# class TargetStockfishBoardLoader(TargetContextBoardsLoader):
#     def __init__(self, path_stockfish, *args, **kwargs):
#         super(TargetStockfishBoardLoader, self).__init__(*args, **kwargs)
#         self.engine = chess.engine.SimpleEngine.popen_uci(path_stockfish)

#     def gen_pairs(self):
#         target = []
#         context = []
#         negatives = []

#         for _ in range(self.PAIR_CNT - 5):
#             # Берём случайную доску
#             i = random.randint(0, len(self.encoded_boards) - 1)
#             target.append(self.encoded_boards[i])

#             # Генерация контекста
#             start = max(0, i - self.WINDOW_SIZE)
#             end = min(len(self.encoded_boards), i + self.WINDOW_SIZE + 1)
#             j = random.choice(list(range(start, i)) + list(range(i + 1, end)))
#             context.append(self.encoded_boards[j])

#             # Генерация негативных примеров из пула self.pool_encoded_boards
#             neg_samples = random.sample(self.pool_encoded_boards, self.NEGATIVES_CNT)
#             negatives.extend(neg_samples)

#         # Добавляем дополнительный контекст, связанный со стокфишем
#         i = random.randint(0, len(self.encoded_boards) - 1)
#         info = self.engine.analyse(self.boards[i], limit=chess.engine.Limit(depth=16))
#         if 'pv' in info:
#             moves = info['pv']
#             prev_board = self.boards[i].copy()
#             for move in moves:
#                 prev_board.push(move)
#                 target.append(self.encoded_boards[i])
#                 context.append(self.board_encoder.encode(prev_board, output_type='numpy'))
#                 neg_samples = random.sample(self.pool_encoded_boards, self.NEGATIVES_CNT)
#                 negatives.extend(neg_samples)

#         return (
#             self.board_encoder.make_batch(target),
#             self.board_encoder.make_batch(context),
#             self.board_encoder.make_batch(negatives)
#         )


class BoardTargetDataloader:
    def __init__(
        self, data: pd.DataFrame,
        board_encoder,
        device,
        batch_size: int = 128
    ):
        self.data = data.copy()
        self.data['moves'] = self.data['moves'].str.split()
        self.board_encoder = board_encoder
        self.batch_size = batch_size
        self._indices = np.random.permutation(len(data))
        self._current = 0
        self.buffer = []
        # Calculate total number of moves across all games
        self.length = self.data['moves'].apply(len).sum()
        self.device = device

    def __len__(self):
        return int(self.length) // int(self.batch_size)

    def __iter__(self):
        self._indices = np.random.permutation(len(self.data))
        self._current = 0
        self.buffer = []
        return self

    def __next__(self):
        while len(self.buffer) < self.batch_size:
            if self._current >= len(self.data):
                if not self.buffer:
                    raise StopIteration
                else:
                    break  # Return remaining elements
            idx = self._indices[self._current]
            self._current += 1
            row = self.data.iloc[idx]
            moves = row['moves']
            try:
                start, end = map(int, row['marks'].split(','))
            except (AttributeError, ValueError):
                start = end = 0
            board = chess.Board()
            party_boards = []
            for i, move_str in enumerate(moves):
                move = chess.Move.from_uci(move_str)
                board.push(move)
                encoded = self.board_encoder.encode(board)
                label = 1 if start <= i < end else 0
                party_boards.append((encoded, label))
            self.buffer.extend(party_boards)

        batch_size = min(self.batch_size, len(self.buffer))
        batch_data = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]

        features = np.stack([d[0] for d in batch_data])
        labels = np.array([d[1] for d in batch_data])

        return (
            torch.tensor(features, device=self.device),
            torch.tensor(labels, device=self.device)
        )