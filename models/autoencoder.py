import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.ReLU(),
            nn.Linear(12,6)
        )

        self.decoder = nn.Sequential(
            nn.Linear(6,12),
            nn.ReLU(),
            nn.Linear(12,24),
            nn.ReLU(),
            nn.Linear(24,   input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction