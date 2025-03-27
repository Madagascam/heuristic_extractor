import torch
import torch.nn as nn

# class Board2Vec(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int,
#             output_dim: int
#         ):
#         super(Board2Vec, self).__init__()

#         # Эмбеддинги
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(hidden_dim)
#         self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(hidden_dim)
#         self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=8, stride=1, padding=0)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(in_features=hidden_dim + 6, out_features=output_dim)

#         self.relu = nn.ReLU()

#     def forward(self, boards: torch.Tensor, adv: torch.Tensor):
#         x = boards.reshape((-1, 1, 8, 8))

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(self.conv3(out))
#         out = torch.flatten(out, 1)

#         y = torch.concat((out, adv), dim=1)
#         y = self.fc(y).squeeze()
#         return y


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
    def __init__(self, hidden_dim, output_dim, input_channel):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channel, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # Резидуальные блоки
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        self.block3 = ResidualBlock(hidden_dim)
        
        # Глобальный пуллинг и финальные слои
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, boards: torch.Tensor):
        # boards: (batch_size, input_channel, 8, 8)
        x = self.initial(boards)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x