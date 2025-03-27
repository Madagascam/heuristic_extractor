import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Union
import matplotlib.pyplot as plt
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ================== Функции для подготовки данных и работы с датасетом ==================

def collate_fn(batch):
    """
    Паддит последовательности в batch до максимальной длины.
    batch — список [(X_seq, y_seq), (X_seq, y_seq), ...].
    Возвращает тензоры (features, targets) размерностей:
    (batch_size, max_len, D) и (batch_size, max_len).
    """
    max_len = max(x.shape[0] for x, _ in batch)
    features, targets = [], []
    for (x_seq, y_seq) in batch:
        # Если y_seq двумерный (n,1), превращаем его в одномерный (n,)
        y_seq = np.array(y_seq).flatten()
        M = x_seq.shape[0]
        pad_size = max_len - M
        x_pad = np.pad(x_seq, ((0, pad_size), (0, 0)), mode='constant')
        y_pad = np.pad(y_seq, (0, pad_size), mode='constant')
        features.append(torch.tensor(x_pad, dtype=torch.float32))
        targets.append(torch.tensor(y_pad, dtype=torch.float32))
    features = torch.stack(features, dim=0)
    targets = torch.stack(targets, dim=0)
    return features, targets

class ChessManyToManyDataset(Dataset):
    """
    Датасет для many-to-many задачи.
    Ожидается, что X и y уже сгруппированы по партиям:
      - X: список numpy-массивов, где каждый массив имеет форму (M, D)
      - y: список numpy-массивов, где каждый массив имеет форму (M,)
    """
    def __init__(self, X_list: Union[List[np.ndarray], pd.DataFrame], y_list: Union[List[np.ndarray], pd.Series]):
        self.X_list = X_list
        self.y_list = y_list

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        return self.X_list[idx], self.y_list[idx]

# ================== Модель BiLSTM для many-to-many ==================

class ChessMoveBiLSTM(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=128, num_layers=2):
        """
        BiLSTM для many-to-many задачи. На входе (batch_size, seq_len, input_dim).
        На выходе (batch_size, seq_len) — логиты.
        """
        super(ChessMoveBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Выдаём логит на каждый шаг
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
        #                     batch_first=True)
        # self.fc = nn.Linear(hidden_dim, 1)  # Выдаём логит на каждый шаг

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)         # (batch_size, seq_len, hidden_dim*2)
        logits = self.fc(lstm_out)         # (batch_size, seq_len, 1)
        return logits.squeeze(-1)          # (batch_size, seq_len)

# ================== Функции обучения и валидации ==================

def train_model(model, train_loader, test_loader,
                num_epochs=100, lr=0.001, pos_weight_val=1.0):
    """
    Обучение модели BiLSTM с использованием BCEWithLogitsLoss с модификацией штрафа для 1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # убираем агрегацию в BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # Применяем вес только к тем элементам, где y_batch == 1
            loss = loss * (y_batch * pos_weight_val + (1 - y_batch))  # штрафуем только за 1

            # Суммируем потери
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_acc, precision, recall = validate_model(model, test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {total_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    return model


def validate_model(model, test_loader):
    """
    Валидация модели на test_loader.
    Возвращает (avg_loss, accuracy, precision, recall).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    TP = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            # Применяем сигмоиду и порог 0.5 для получения предсказаний
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()

            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()

            # Вычисляем True Positives, False Positives, False Negatives
            TP += ((predictions == 1) & (y_batch == 1)).sum().item()
            FP += ((predictions == 1) & (y_batch == 0)).sum().item()
            FN += ((predictions == 0) & (y_batch == 1)).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return avg_loss, accuracy, precision, recall


def plot_predictions(model, X_seq, y_seq):
    """
    Визуализирует предсказания модели для одной партии.
    X_seq: np.ndarray, pd.DataFrame или torch.Tensor формы (M, D) – признаки партии.
    y_seq: массив истинных меток формы (M,) или (M, 1).
           Если значение равно 1, точка будет красной, если 0 – синей.
    """
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Преобразуем X_seq в тензор, если он не tensor
    if hasattr(X_seq, 'values'):
        X_seq = X_seq.values
    if not isinstance(X_seq, torch.Tensor):
        X_seq = torch.tensor(X_seq, dtype=torch.float32)

    # Преобразуем y_seq в numpy-массив, если он DataFrame/Series или tensor
    if hasattr(y_seq, 'values'):
        y_seq = y_seq.values
    if isinstance(y_seq, torch.Tensor):
        y_seq = y_seq.cpu().numpy()
    # Если y_seq имеет форму (M,1), преобразуем в (M,)
    y_seq = y_seq.flatten()

    # Добавляем размер батча: (1, M, D)
    X_seq = X_seq.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X_seq).squeeze(0)  # (M,)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Определяем цвета: если истинная метка равна 1, цвет красный, иначе синий.
    colors = ['red' if target == 1 else 'blue' for target in y_seq]

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(probs)), probs, label='Predicted Probability', color='gray', linestyle='--')
    plt.scatter(range(len(probs)), probs, c=colors, label='True Label')
    plt.xlabel('Move Number')
    plt.ylabel('Predicted Probability')
    plt.title('Model Predictions with True Labels')
    plt.legend()
    plt.show()


def plot_random_games(model, X_list, y_list, n_games=3):
    """
    Отображает графики предсказаний для случайно выбранных партий.

    Аргументы:
      model: обученная модель.
      X_list: список партий, каждая партия имеет форму (M, D) - признаки.
      y_list: список меток для партий, каждая партия имеет форму (M,) или (M, 1).
      n_games: число случайных партий для отображения.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Выбираем случайные индексы
    indices = random.sample(range(len(X_list)), min(n_games, len(X_list)))

    fig, axs = plt.subplots(n_games, 1, figsize=(10, 5 * n_games))
    # Если отображается только одна партия, делаем axs списком
    if n_games == 1:
        axs = [axs]

    for ax, idx in zip(axs, indices):
        X_seq = X_list[idx]
        y_seq = y_list[idx]

        # Если данные представлены в DataFrame или Series, преобразуем их в numpy
        if hasattr(X_seq, 'values'):
            X_seq = X_seq.values
        if hasattr(y_seq, 'values'):
            y_seq = y_seq.values

        # Если не тензор, преобразуем в tensor
        if not isinstance(X_seq, torch.Tensor):
            X_seq = torch.tensor(X_seq, dtype=torch.float32)
        if isinstance(y_seq, torch.Tensor):
            y_seq = y_seq.cpu().numpy()
        # Приводим y_seq к одномерному виду
        y_seq = y_seq.flatten()

        # Добавляем измерение батча: (1, M, D)
        X_seq_batch = X_seq.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(X_seq_batch).squeeze(0)  # (M,)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Цвет точек определяется истинной меткой: 1 -> красный, 0 -> синий
        colors = ['red' if label == 1 else 'blue' for label in y_seq]

        ax.plot(range(len(probs)), probs, label='Predicted Probability', color='gray', linestyle='--')
        ax.scatter(range(len(probs)), probs, c=colors, label='True Label')
        ax.set_xlabel('Move Number')
        ax.set_ylabel('Predicted Probability')
        ax.set_title(f'Game {idx}')
        ax.legend()

    plt.tight_layout()
    plt.show()
# ================== Основной блок: загрузка данных, обучение, валидация, визуализация ==================

if __name__ == '__main__':
    # Загружаем данные из файлов
    X_train, y_train = torch.load('D:/heuristic_extractor/board2vec/train.pth')
    X_test, y_test = torch.load('D:/heuristic_extractor/board2vec/test.pth')

    # Для проверки печатаем первые несколько строк
    print("Первые 3 партии из X_train:")
    print(X_train[:3])
    print("Первые 3 партии из y_train:")
    print(y_train[:3])

    # Предполагаем, что данные уже сгруппированы по партиям и представлены как списки или аналогичные структуры.
    # Если они хранятся в виде DataFrame/Series, то их можно оставить как есть, если они уже имеют нужную форму.
    train_dataset = ChessManyToManyDataset(X_train, y_train)
    test_dataset = ChessManyToManyDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Создаём модель
    model = ChessMoveBiLSTM(input_dim=66, hidden_dim=128, num_layers=2)

    # Обучаем модель. Здесь pos_weight_val=5.0 - значение подбирается эмпирически для компенсации дисбаланса.
    trained_model = train_model(model, train_loader, test_loader,
                                num_epochs=100, lr=0.001, pos_weight_val=10)

    # Финальная валидация
    val_loss, val_acc, precision, recall = validate_model(trained_model, test_loader)
    print(f"Final Val Loss: {val_loss:.4f}, Final Val Acc: {val_acc:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Визуализируем предсказания для первой партии тестового набора
    plot_random_games(trained_model, X_test, y_test, n_games=6)
