import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import time
import h5py
from ..utils.dataloader import TargetContextBoardsLoader
from ..utils.board_encoder import MatrixEncoder
from .config import *
from .model import Board2Vec, WrapperNet
from torch.utils.data import Dataset, DataLoader

try:
    import torch_directml
except ImportError:
    torch_directml = None

# Определение устройства
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch_directml is not None:
    device = torch_directml.device()
else:
    device = torch.device("cpu")
print(f'device: {device}')

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, device):
        self.file = h5py.File(h5_file, 'r')
        self.positions = self.file['position']
        self.targets = self.file['target']
        self.device = device

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        pos = torch.tensor(self.positions[idx], device=self.device)
        tar = torch.tensor(self.targets[idx], device=self.device)
        return pos, tar

dataset = HDF5Dataset(data_path, device)

import torch

def bce_with_logits_loss(input: torch.Tensor, target: torch.Tensor):
    max_val = torch.clamp(-input, min=0)
    loss = input - input * target + max_val + torch.log(torch.exp(-max_val) + torch.exp(-input - max_val))
    return loss.mean()

def run_train():
    # Модель
    model = WrapperNet(hidden_dim, output_dim).to(device)
    weight_path = weight_dir + 'CNN_1.pth'
    
    # Загрузка весов с учетом устройства
    if os.path.exists(weight_path):
        print('loading weights...')
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    # Даталоадер
    dataloader = DataLoader(dataset, batch_size=128)

    # Оптимизатор
    optimizer = optim.Adadelta(model.parameters())

    # Планировщик
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Цикл обучения
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()
        for boards, targets in dataloader:
            output = model(boards)
            loss = bce_with_logits_loss(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}, "
              f"lr: {scheduler.get_last_lr()}, time: {end_time-start_time:.2f}")

        torch.save(model.state_dict(), weight_dir + f'CNN_{epoch + 1}_{int(total_loss)}.pth')