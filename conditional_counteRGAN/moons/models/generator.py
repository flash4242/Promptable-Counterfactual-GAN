import torch.nn as nn
import torch

class ResidualGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, input_dim)
        )

    def forward(self, x, target):
        x_cat = torch.cat([x, target], dim=1)
        residual = self.net(x_cat)
        return residual
