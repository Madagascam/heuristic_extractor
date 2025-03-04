import torch
import chess
from .model import Board2Vec
from .config import weight_dir
from ..utils.board_encoder import SparseEncoder

# load_path - путь к весам обученной модели
model = Board2Vec()
model.load_state_dict(torch.load(weight_dir + 'MLP_1.pth', weights_only=True))
model.eval()
encoder = SparseEncoder()


def board2vec(board: chess.Board):
    with torch.no_grad():
        encoded = encoder.encode(board, output_type='torch')
        return model(encoded)


if __name__ == '__main__':
    print(board2vec(chess.Board))