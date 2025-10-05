import torch.nn as nn
import torch

class ResidualGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes + input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, input_dim)
        )

    def forward(self, x, target_onehot, mask=None):
        h = torch.cat([x, target_onehot, mask], dim=1)
        raw_residual = self.net(h)
        masked_residual = raw_residual * mask
        return raw_residual, masked_residual