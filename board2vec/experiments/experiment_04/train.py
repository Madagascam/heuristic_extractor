import torch
import torch.optim as optim
import os
import time
import pandas as pd
from ..utils.board_encoder import MatrixEncoder
from ..utils.dataloader import BoardTargetDataloader
from .config import (
    hidden_dim,
    output_dim,
    learning_rate,
    weight_dir,
    step_size,
    gamma,
    num_epochs
)
from .model import WrapperNet

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


# Функция для постепенной загрузки данных
def load_data_in_chunks(file_path, chunk_size):
    data = pd.read_csv(file_path, chunksize=chunk_size)
    for chunk in data:
        yield chunk


# Определение функции потерь
def bce_loss(probs: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-7):
    """
    Binary Cross-Entropy Loss для готовых вероятностей.

    Args:
        probs (torch.Tensor): Предсказанные вероятности.
        target (torch.Tensor): Истинные метки (тензор значений 0 или 1).
        epsilon (float): Маленькое значение для стабильности вычислений.

    Returns:
        torch.Tensor: Среднее значение BCE по всем элементам.
    """
    probs = torch.clamp(probs, min=epsilon, max=1 - epsilon)
    loss = -(target * torch.log(probs) + (1 - target) * torch.log(1 - probs))
    return loss.mean()


encoder = MatrixEncoder(color='each', meta=False)


def run_train():
    # Модель
    model = WrapperNet(hidden_dim, output_dim, input_channel=encoder.get_encoded_shape()[0]).to(device)
    weight_path = weight_dir + 'CNN_E.pth'

    # Загрузка весов с учетом устройства
    if os.path.exists(weight_path):
        print('loading weights...')
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)

    # Оптимизатор
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    # Планировщик
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Цикл обучения
    chunk_size = 500  # Размер части данных, загружаемой за одну эпоху
    train_data_path = 'D:/Program Files/JupyterLabWorkspace/chess_data/splits/train.csv'
    data_generator = load_data_in_chunks(train_data_path, chunk_size)
    for epoch in range(num_epochs):
        start_time = time.time()

        # Загрузка валидационного и тестового датасетов
        val_data = pd.read_csv('D:/Program Files/JupyterLabWorkspace/chess_data/splits/val.csv')
        test_data = pd.read_csv('D:/Program Files/JupyterLabWorkspace/chess_data/splits/test.csv')

        val_dataloader = BoardTargetDataloader(
            data=val_data,
            board_encoder=encoder,
            device=device,
            batch_size=128
        )

        test_dataloader = BoardTargetDataloader(
            data=test_data,
            board_encoder=encoder,
            device=device,
            batch_size=128
        )

        # Загрузка части тренировочных данных
        chunk = next(data_generator, None)
        if chunk is None:
            raise ValueError("No more data chunks available.")

        dataloader = BoardTargetDataloader(
            data=chunk,
            board_encoder=encoder,
            device=device,
            batch_size=128
        )

        cnt = 0
        for boards, targets in dataloader:
            print(f'{(cnt := cnt + 1)}/{len(dataloader)}', end='\r')
            output = model(boards)
            loss = bce_loss(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        end_time = time.time()

        # Валидация
        model.eval()
        val_loss = 0
        val_cnt = 0
        with torch.no_grad():
            for boards, targets in val_dataloader:
                output = model(boards)
                val_loss += bce_loss(output, targets).item()
                val_cnt += 1
        val_loss /= val_cnt
        model.train()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Val Loss: {val_loss}, lr: {scheduler.get_last_lr()}, "
              f"time: {end_time-start_time:.2f}")

        torch.save(model.state_dict(),
                   weight_dir + f'CNN_{epoch + 1}_{(val_loss):.4f}.pth')

    # Тестирование
    model.eval()
    test_loss = 0
    test_cnt = 0
    with torch.no_grad():
        for boards, targets in test_dataloader:
            output = model(boards)
            test_loss += bce_loss(output, targets).item()
            test_cnt += 1
    test_loss /= test_cnt
    print(f"Test Loss: {test_loss}")