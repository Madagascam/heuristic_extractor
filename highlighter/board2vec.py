import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for stable deep network training.
    Architecture: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input/output channels (keeps dimensions same)
        """
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Activation function (inplace saves memory)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connection
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Output tensor with same shape as input
        """
        residual = x  # Save original input for skip connection
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection and apply final activation
        out += residual
        out = self.relu(out)
        
        return out

class Board2Vec(nn.Module):
    """
    Neural network for converting chess board representation to embedding vector.
    Architecture:
    - Initial convolution
    - Multiple residual blocks
    - Global average pooling
    - Fully connected layers with dropout
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        """
        Args:
            hidden_dim: Number of channels in hidden layers
            output_dim: Dimension of output embedding
        """
        super().__init__()
        
        # Input block processes 12-channel board representation:
        # (6 pieces × 2 colors + castling rights + en passant + promotion info)
        self.initial = nn.Sequential(
            nn.Conv2d(12, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Stack of residual blocks for deep feature extraction
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        self.block3 = ResidualBlock(hidden_dim)

        # Global pooling reduces spatial dimensions to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final fully connected layers with dropout for regularization
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(hidden_dim, output_dim)  # Final projection to target dimension
        )

    def forward(self, boards: Tensor) -> Tensor:
        """
        Forward pass converting board representation to embedding
        Args:
            boards: Input tensor of shape (batch_size, 12, 8, 8)
                    representing chess boards
        Returns:
            Embedding vectors of shape (batch_size, output_dim)
        """
        # Initial feature extraction
        x = self.initial(boards)  # (batch_size, hidden_dim, 8, 8)
        
        # Process through residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global pooling and flatten
        x = self.avgpool(x)  # (batch_size, hidden_dim, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, hidden_dim)
        
        # Final projection to embedding space
        x = self.fc(x)  # (batch_size, output_dim)
        
        return x