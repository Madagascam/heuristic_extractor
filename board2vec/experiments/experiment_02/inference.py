import torch
import torch_directml
import chess
import numpy as np
from .model import Board2Vec
from .config import hidden_dim, output_dim, weight_dir
from ..utils.board_encoder import MatrixEncoder
from typing import List

# Определение устройства
device = torch.device('cpu')
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch_directml is not None:
#     device = torch_directml.device()
# else:
#     device = torch.device("cpu")
print(f'device: {device}')

# load_path - путь к весам обученной модели
model = Board2Vec(hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load(weight_dir + 'CNN_3.pth', map_location=device))
model.eval()
encoder = MatrixEncoder()


def board2vec(boards: List[chess.Board]):
    with torch.no_grad():
        targets = torch.tensor(np.array([encoder.encode(board) for board in boards]), device=device)
        return model(targets).squeeze().cpu().detach().numpy()


if __name__ == '__main__':
    print(board2vec(chess.Board))