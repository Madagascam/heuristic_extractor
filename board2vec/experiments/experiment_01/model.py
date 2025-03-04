import torch.nn as nn


class Board2Vec(nn.Module):
    def __init__(
            self,
            input_dim: int,
            sparse_hidden: int,
            hidden_dim: int,
            output_dim: int
        ):
        super(Board2Vec, self).__init__()

        # Эмбеддинги
        layers = [
            nn.Linear(input_dim, sparse_hidden),
            nn.ReLU(),
            nn.Linear(sparse_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ]
        self.embedding = nn.Sequential(*layers)

    def forward(self, x):
        return self.embedding(x)