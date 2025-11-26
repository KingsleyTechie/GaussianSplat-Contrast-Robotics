import torch
import torch.nn as nn

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=9, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z
