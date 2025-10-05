import torch.nn as nn
import torch
import torch.nn.utils.spectral_norm as spectral_norm


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim + num_classes, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim//2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim//2, hidden_dim//2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim//2, 1))

        )

    def forward(self, x, target_onehot):
        h = torch.cat([x, target_onehot], dim=1)
        return self.net(h)
