import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Board2Vec(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        # Входной блок с 12 каналов (6 фигур × 2 цвета)
        self.initial = nn.Sequential(
            nn.Conv2d(12, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Резидуальные блоки
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)

        # Глобальный пуллинг и финальные слои
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),  # +6 дополнительных признаков
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, boards: torch.Tensor, adv: torch.Tensor):
        # boards: (batch_size, 12, 8, 8)
        x = self.initial(boards)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Конкатенация с дополнительными признаками
        x = torch.cat([x, adv], dim=1)
        x = self.fc(x)
        return x