import torch
try:
    import torch_directml
except ImportError:
    torch_directml = None
import numpy as np
import torch.optim as optim
import pandas as pd
import os
import time
from ..utils.dataloader import TargetContextBoardsLoader
from ..utils.board_encoder import MatrixEncoder
from .config import *
from .model import Board2Vec

# Определение устройства
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch_directml is not None:
    device = torch_directml.device()
else:
    device = torch.device("cpu")
print(f'device: {device}')

data = pd.read_csv(data_path, header=0)
games_series = data['moves'].str.strip().str.split()

def criterion(target_embed: torch.Tensor, context_embed: torch.Tensor, negatives_embed: torch.Tensor):
    # Положительные примеры: скалярное произведение между target и context
    pos_scores = torch.mul(target_embed, context_embed).sum(dim=1)
    # Замена logsigmoid на его математический эквивалент
    pos_loss = -(-torch.log(1 + torch.exp(-pos_scores)))

    # Негативные примеры: скалярное произведение между target и negatives
    neg_scores = torch.bmm(negatives_embed, target_embed.unsqueeze(2)).squeeze(2)
    # Замена logsigmoid на его математический эквивалент
    neg_loss = -(-torch.log(1 + torch.exp(neg_scores))).sum(dim=1)

    # Общая потеря: усредняем по батчу
    loss = (pos_loss + neg_loss).mean()
    return loss

def run_train():
    # Модель
    model = Board2Vec(hidden_dim, output_dim).to(device)
    weight_path = weight_dir + 'CNN_2.pth'
    
    # Загрузка весов с учетом устройства
    # if os.path.exists(weight_path):
    #     print('loading weights...')
    #     state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    #     model.load_state_dict(state_dict)
    model = model.to(device)

    # Даталоадер
    matrix_encoder = MatrixEncoder()
    dataloader = TargetContextBoardsLoader(
        games_series,
        board_encoder=matrix_encoder,
        window_size=WINDOW_SIZE,
        game_count=GAME_COUNT,
        pair_cnt=PAIR_CNT,
        subset_size=SUBSET_SIZE,
        negatives_cnt=NEGATIVES_COUNT,
        device=device
    )

    # Оптимизатор
    optimizer = optim.Adadelta(model.parameters())

    # Планировщик
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Цикл обучения
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()
        for target, context, negatives in dataloader:
            target_embed = model(target)
            context_embed = model(context)
            negatives_embed = model(negatives.reshape((-1, *matrix_encoder.get_encoded_shape()))).reshape((-1, NEGATIVES_COUNT, output_dim))
            loss = criterion(target_embed, context_embed, negatives_embed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}, "
              f"lr: {scheduler.get_last_lr()}, time: {end_time-start_time:.2f}")

        if (epoch + 1) % 10 == 0:
            # Сохраняем веса модели
            torch.save(model.state_dict(), weight_dir + f'CNN_{(epoch + 1) // 10}_{int(total_loss)}.pth')