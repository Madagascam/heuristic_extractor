import torch
import chess
from .model import Board2Vec
# from .config import weight_dir
from ..utils.board_encoder import SimpleEncoder

# load_path - путь к весам обученной модели
model = Board2Vec(hidden_dim=48)
# model.load_state_dict(torch.load(weight_dir + 'MLP_1.pth', weights_only=True))
model.eval()
encoder = SimpleEncoder()


def board2vec(board: chess.Board):
    with torch.no_grad():
        board, adv = encoder.encode(board, output_type='torch')
        return model(board.reshape((1, 1, 8, 8)), adv.reshape((1, 6))).detach().numpy()


if __name__ == '__main__':
    print(board2vec(chess.Board))