import torch
import h5py
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'r')
        self.boards = self.h5_file['boards']
        self.targets = self.h5_file['targets']

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = torch.from_numpy(self.boards[idx])
        target = torch.from_numpy(self.targets[idx])
        return board, target