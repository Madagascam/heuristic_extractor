import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
<<<<<<< HEAD
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
=======
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
>>>>>>> caf572674f5038347f87e1d23ee79ab4ed48aee6
        self.relu = nn.ReLU(inplace=True)

        # Для случая изменения размера или количества каналов
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Board2Vec(nn.Module):
    def __init__(self, hidden_dim, output_dim, input_channel):
        super().__init__()
        # Входной блок с увеличением канальности
        self.initial = nn.Sequential(
            nn.Conv2d(input_channel, hidden_dim, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Блоки с уменьшением размерности через stride
        self.block1 = ResidualBlock(hidden_dim, hidden_dim, stride=1)
        self.block2 = ResidualBlock(hidden_dim, hidden_dim*2, stride=2)
        self.block3 = ResidualBlock(hidden_dim*2, hidden_dim*2, stride=1)
        self.block4 = ResidualBlock(hidden_dim*2, hidden_dim*4, stride=2)
        self.block5 = ResidualBlock(hidden_dim*4, hidden_dim*4, stride=1)
        self.block6 = ResidualBlock(hidden_dim*4, hidden_dim*4, stride=1)

        # Глобальный пуллинг и финальные слои
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, output_dim)
        )

    def forward(self, boards: torch.Tensor):
        x = self.initial(boards)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class WrapperNet(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, input_channel):
        super().__init__()
        self.board2vec = Board2Vec(hidden_dim, embedding_dim, input_channel)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, boards: torch.Tensor):
        x = self.board2vec(boards)
        x = self.fc(x)
        return x.squeeze()