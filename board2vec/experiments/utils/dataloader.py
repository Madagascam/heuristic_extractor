import chess
import chess.engine
import random
import chess.engine
import torch
import numpy as np
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
        self.games = games
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

    def check_game(self, game_idx):
        flag = True
        game = self.games[game_idx]
        for i, move in enumerate(game):
            try:
                chess.Move.from_uci(move)
            except chess.InvalidMoveError:
                if move == '' and i == len(game) - 1:
                    del self.games[game_idx][i]
                else:
                    flag = False
        return flag

    def update_board_pool(self):
        """Обновляет пул досок, выбирая случайное подмножество партий."""
        for i in range(self.SUBSET_SIZE):
            idx = random.randint(0, len(self.games) - 1)
            while not self.check_game(idx):
                idx = random.randint(0, len(self.games) - 1)
            board = chess.Board()
            for move in self.games[idx]:
                board.push(chess.Move.from_uci(move))
                self.pool_encoded_boards[i] = self.board_encoder.encode(board)

    def set_game(self):
        cur_game = random.randint(0, len(self.games) - 1)
        while not self.check_game(cur_game):
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
        target = np.empty((self.PAIR_CNT, *self.board_encoder.get_encoded_shape()), dtype=np.float32, order='C')
        context = np.empty((self.PAIR_CNT, *self.board_encoder.get_encoded_shape()), dtype=np.float32, order='C')
        negatives = np.empty((self.PAIR_CNT, self.NEGATIVES_CNT, *self.board_encoder.get_encoded_shape()), dtype=np.float32, order='C')

        for idx in range(self.PAIR_CNT):
            i = random.randint(0, len(self.encoded_boards) - 1)
            target[idx] = self.encoded_boards[i]

            # Генерация контекста
            start = max(0, i - self.WINDOW_SIZE)
            end = min(len(self.encoded_boards), i + self.WINDOW_SIZE + 1)
            j = random.choice(list(range(start, i)) + list(range(i + 1, end)))
            context[idx] = self.encoded_boards[j]

            # Генерация негативных примеров из пула self.pool_encoded_boards
            indices = np.random.choice(len(self.pool_encoded_boards), size=self.NEGATIVES_CNT, replace=False)
            negatives[idx] = self.pool_encoded_boards[indices]

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


class TargetStockfishBoardLoader(TargetContextBoardsLoader):
    def __init__(self, path_stockfish, *args, **kwargs):
        super(TargetStockfishBoardLoader, self).__init__(*args, **kwargs)
        self.engine = chess.engine.SimpleEngine.popen_uci(path_stockfish)

    def gen_pairs(self):
        target = []
        context = []
        negatives = []

        for _ in range(self.PAIR_CNT - 5):
            # Берём случайную доску
            i = random.randint(0, len(self.encoded_boards) - 1)
            target.append(self.encoded_boards[i])

            # Генерация контекста
            start = max(0, i - self.WINDOW_SIZE)
            end = min(len(self.encoded_boards), i + self.WINDOW_SIZE + 1)
            j = random.choice(list(range(start, i)) + list(range(i + 1, end)))
            context.append(self.encoded_boards[j])

            # Генерация негативных примеров из пула self.pool_encoded_boards
            neg_samples = random.sample(self.pool_encoded_boards, self.NEGATIVES_CNT)
            negatives.extend(neg_samples)

        # Добавляем дополнительный контекст, связанный со стокфишем
        i = random.randint(0, len(self.encoded_boards) - 1)
        info = self.engine.analyse(self.boards[i], limit=chess.engine.Limit(depth=16))
        if 'pv' in info:
            moves = info['pv']
            prev_board = self.boards[i].copy()
            for move in moves:
                prev_board.push(move)
                target.append(self.encoded_boards[i])
                context.append(self.board_encoder.encode(prev_board, output_type='numpy'))
                neg_samples = random.sample(self.pool_encoded_boards, self.NEGATIVES_CNT)
                negatives.extend(neg_samples)

        return (
            self.board_encoder.make_batch(target),
            self.board_encoder.make_batch(context),
            self.board_encoder.make_batch(negatives)
        )
