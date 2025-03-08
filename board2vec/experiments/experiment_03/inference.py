import torch
import chess
from .model import Board2Vec
from .config import hidden_dim, output_dim, weight_dir
from ..utils.board_encoder import MatrixEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# load_path - путь к весам обученной модели
model = Board2Vec(hidden_dim, output_dim)
model.load_state_dict(torch.load(weight_dir + 'CNN_1.pth', weights_only=True, map_location=device))
model.eval()
encoder = MatrixEncoder()


def board2vec(board: chess.Board):
    with torch.no_grad():
        target = encoder.encode(board, output_type='torch')
        target = encoder.make_batch([target])
        return model(*target).squeeze().detach().numpy()


if __name__ == '__main__':
    print(board2vec(chess.Board))