import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
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
        # Входной блок с 12 каналами
        self.initial = nn.Sequential(
            nn.Conv2d(12, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # Резидуальные блоки
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.block3 = ResidualBlock(hidden_dim)
        self.block4 = ResidualBlock(hidden_dim)

        # Глобальный пуллинг и финальные слои
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, boards: torch.Tensor):
        # boards: (batch_size, 12, 8, 8)
        x = self.initial(boards)
        x = self.block1(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class WrapperNet(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.board2vec = Board2Vec(hidden_dim, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, boards: torch.Tensor):
        x = self.board2vec(boards)
        x = self.fc(x)
        return x.squeeze()