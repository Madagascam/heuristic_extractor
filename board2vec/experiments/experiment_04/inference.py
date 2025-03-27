import torch
import torch_directml
import chess
import numpy as np
from .model import WrapperNet
from .config import hidden_dim, output_dim, weight_dir
from ..utils.board_encoder import MatrixEncoder
from typing import List

cpu = torch.device('cpu')
device = torch_directml.device()
# print(f'device: {device}')

encoder = MatrixEncoder(color='each', meta=False)

wrapper = WrapperNet(hidden_dim, output_dim, encoder.get_encoded_shape()[0])
wrapper.load_state_dict(torch.load(weight_dir + 'CNN_E.pth', map_location=cpu))
model = wrapper.board2vec.to(device)
model.eval()

wrapper.to(device)
wrapper.eval()


def board2vec(boards: List[chess.Board]):
    with torch.no_grad():
        targets = torch.tensor(
            np.array([encoder.encode(board) for board in boards])
        ).to(device)
        return model(targets).squeeze().detach().cpu().numpy()


def is_interesting(boards: List[chess.Board]):
    with torch.no_grad():
        targets = torch.tensor(
            np.array([encoder.encode(board) for board in boards])
        ).to(device)
        return wrapper(targets).squeeze().detach().cpu().numpy()


if __name__ == '__main__':
    print(board2vec(chess.Board))