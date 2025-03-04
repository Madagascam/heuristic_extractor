import chess
import random
import numpy as np
import torch
from typing import List
from .board_encoder import BoardEncoder

class TargetContextBoardsLoader:
    def __init__(
            self, games: List[List[str]],
            board_encoder: BoardEncoder,
            window_size: int,
            game_count: int,
            pair_cnt: int,
            subset_size: int
        ):
        self.games = games
        self.board_encoder = board_encoder
        self.WINDOW_SIZE = window_size
        self.PAIR_CNT = pair_cnt
        self.GAME_COUNT = game_count
        self.SUBSET_SIZE = subset_size  # Размер подмножества партий для пула

        self.games_left = self.GAME_COUNT
        self.all_boards = []  # Пул досок из подмножества
        self.boards = []  # Доски текущей игры

        # Инициализируем пул досок в первый раз
        self.update_board_pool()

    def check_game(self, game):
        flag = True
        for move in game:
            try:
                chess.Move.from_uci(move)
            except chess.InvalidMoveError:
                flag = False
        return flag

    def update_board_pool(self):
        """Обновляет пул досок, выбирая случайное подмножество партий."""
        self.all_boards = []
        cnt = self.SUBSET_SIZE
        while cnt > 0:
            idx = random.randint(0, len(self.games) - 1)
            if not self.check_game(self.games[idx]):
                continue
            board = chess.Board()
            for move in self.games[idx]:
                board.push(chess.Move.from_uci(move))
                self.all_boards.append(self.board_encoder.encode(board, output_type='numpy'))
            cnt -= 1

    def set_game(self):
        cur_game = random.randint(0, len(self.games) - 1)
        while not self.check_game(self.games[cur_game]):
            cur_game = random.randint(0, len(self.games) - 1)
        self.games_left -= 1

        self.boards = []
        prev_board = chess.Board()
        game = self.games[cur_game]
        for move in game:
            prev_board.push(chess.Move.from_uci(move))
            encoded = self.board_encoder.encode(prev_board, output_type='numpy')
            self.boards.append(encoded)
        self.boards = np.array(self.boards)
    
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

        target = np.array(target)
        context = np.array(context)
        negatives = np.array(negatives)

        return (torch.tensor(target),
                torch.tensor(context),
                torch.tensor(negatives))

    def __iter__(self):
        return self

    def __next__(self):
        if self.games_left <= 0:
            self.games_left = self.GAME_COUNT
            raise StopIteration
        
        self.set_game()
        
        return self.gen_pairs()