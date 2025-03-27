from experiments.experiment_04.inference import board2vec
from experiments.utils.vizualize import main
import pandas as pd
import chess

# Таблицу можно скачать здесь: https://disk.yandex.ru/d/q1G0W-taR0TSxw
data_path = 'C:/Users/matvey/Documents/chess_data/full_labeled.csv'

data = pd.read_csv(data_path)
boards = []
for game in data['moves'].str.split().head(100):
    board = chess.Board()
    for move in game:
        board.push(chess.Move.from_uci(move))
        boards.append(board.copy())

# Визуализируем
main(boards, board2vec(boards))


# from experiments.experiment_02.train import run_train
# run_train()