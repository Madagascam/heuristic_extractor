import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import os
from ..utils.dataloader import TargetContextBoardsLoader
from ..utils.board_encoder import MatrixEncoder
from .config import *
from .model import Board2Vec

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

data = pd.read_csv(data_path, header=0)
games_series = data['moves'].str.split(' ')     

def criterion(target_embed: torch.Tensor, context_embed: torch.Tensor, negatives_embed: torch.Tensor):
    # Положительные примеры: скалярное произведение между target и context
    pos_scores = torch.mul(target_embed, context_embed).sum(dim=1)
    pos_loss = -torch.nn.functional.logsigmoid(pos_scores)

    # Негативные примеры: скалярное произведение между target и negatives
    neg_scores = torch.bmm(negatives_embed, target_embed.unsqueeze(2)).squeeze(2)
    neg_loss = -torch.nn.functional.logsigmoid(-neg_scores).sum(dim=1)

    # Общая потеря: усредняем по батчу
    loss = (pos_loss + neg_loss).mean()
    return loss

def run_train():
    # Модель
    model = Board2Vec(hidden_dim, output_dim).to(device)
    weight_path = weight_dir + 'CNN_1.pth'
    
    # Загрузка весов с учетом устройства
    if os.path.exists(weight_path):
        print('loading weights...')
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    # Даталоадер
    dataloader = TargetContextBoardsLoader(
        games_series,
        board_encoder=MatrixEncoder(),
        window_size=WINDOW_SIZE,
        game_count=GAME_COUNT,
        pair_cnt=PAIR_CNT,
        subset_size=SUBSET_SIZE,
        negatives_cnt=NEGATIVES_COUNT
    )

    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Планировщик
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Цикл обучения
    for epoch in range(num_epochs):
        total_loss = 0
        for target, context, negatives in dataloader:
            # Перенос данных на устройство
            target = tuple(t.to(device) for t in target)
            context = tuple(c.to(device) for c in context)
            negatives = tuple(n.to(device) for n in negatives)

            target_embed = model(*target)
            context_embed = model(*context)
            negatives_embed = model(*negatives).reshape((-1, NEGATIVES_COUNT, output_dim))
            loss = criterion(target_embed, context_embed, negatives_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Обновляем пул досок после каждой эпохи
        dataloader.update_board_pool()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}, lr: {scheduler.get_last_lr()}")
        
        if (epoch + 1) % 10 == 0:
            # Сохраняем веса модели
            torch.save(model.state_dict(), weight_path)