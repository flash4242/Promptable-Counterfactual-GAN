import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    """Feature-wise linear modulation (mask-aware conditioning)."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta  = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h, cond):
        g = self.gamma(cond)
        b = self.beta(cond)
        return g * h + b


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.film = FiLM(hidden_dim, cond_dim)

    def forward(self, h, cond):
        out = self.fc1(h)
        out = self.bn1(out)
        out = F.relu(self.film(out, cond))
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.film(out, cond)
        return h + out


class ResidualGenerator(nn.Module):
    """
    Generator that:
     - receives x, target_onehot, mask;
     - conditions on (target, mask) via FiLM in each block;
     - enforces hard mask if enabled.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, n_blocks=5, residual_scaling=0.1, enforce_hard_mask=False):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = input_dim + num_classes  # mask + target_onehot
        self.hidden_dim = hidden_dim

        self.fc_in = nn.Linear(input_dim + self.cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, self.cond_dim) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hidden_dim, input_dim)

        self.residual_scaling = float(residual_scaling)
        self.enforce_hard_mask = enforce_hard_mask

    def forward(self, x, target_onehot, mask=None):
        if mask is None:
            mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        mask = mask.to(x.device).to(x.dtype)

        cond = torch.cat([target_onehot, mask], dim=1)
        h = torch.cat([x, cond], dim=1)
        h = F.relu(self.fc_in(h))

        for b in self.blocks:
            h = b(h, cond)

        raw_residual = self.fc_out(h) * self.residual_scaling
        masked_residual = raw_residual * mask

        return raw_residual, masked_residual
