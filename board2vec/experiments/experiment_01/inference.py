import torch
import chess
import numpy as np
from .model import Board2Vec
from .config import weight_dir
from ..utils.board_encoder import SparseEncoder
from .config import input_dim, sparse_hidden, hidden_dim, output_dim
from typing import List

model = Board2Vec(input_dim, sparse_hidden, hidden_dim, output_dim)
model.load_state_dict(torch.load(weight_dir + 'MLP_1.pth', weights_only=True))
model.eval()
encoder = SparseEncoder()


def board2vec(boards: List[chess.Board]):
    with torch.no_grad():
        encoded = torch.tensor(np.array([encoder.encode(board) for board in boards]))
        return model(encoded).detach().numpy()


if __name__ == '__main__':
    print(board2vec(chess.Board))