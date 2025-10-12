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
    Returns:
      - cont_residual: tensor (bs, n_continuous) additive residuals for continuous features
      - cat_logits: dict idx->(bs, n_cat) logits for each categorical feature
      - cat_samples: dict idx->(bs, n_cat) Gumbel-softmax samples (soft/hard depending on `hard`)
    """
    def __init__(self, input_dim, hidden_dim, num_classes, continuous_idx, categorical_info, n_blocks=5, residual_scaling=0.1, tau=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.continuous_idx = list(continuous_idx)
        # categorical_info: dict idx -> {"n": ncat, "raw_values": [...]}
        self.categorical_info = categorical_info

        self.cond_dim = input_dim + num_classes  # mask + target_onehot
        self.fc_in = nn.Linear(input_dim + self.cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, self.cond_dim) for _ in range(n_blocks)])

        # heads
        self.fc_cont = nn.Linear(hidden_dim, len(self.continuous_idx))
        self.fc_cat_logits = nn.ModuleDict({
            str(idx): nn.Linear(hidden_dim, info["n"])
            for idx, info in self.categorical_info.items()
        })

        self.residual_scaling = residual_scaling
        self.tau = tau

    def forward(self, x, target_onehot, mask=None, temperature=None, hard=False):
        device = x.device
        if mask is None:
            mask = torch.ones_like(x, device=device)
        cond = torch.cat([target_onehot.to(device), mask.to(device)], dim=1)
        h = torch.cat([x, cond], dim=1)
        h = F.relu(self.fc_in(h))

        for b in self.blocks:
            h = b(h, cond)

        # continuous residuals
        cont_residual = self.fc_cont(h) * self.residual_scaling  # shape (bs, n_cont)

        # categorical logits & gumbel-softmax samples
        cat_logits = {}
        cat_samples = {}
        tau = self.tau if temperature is None else float(temperature)
        for idx_str, head in self.fc_cat_logits.items():
            logits = head(h)  # (bs, ncat)
            cat_logits[int(idx_str)] = logits
            # Gumbel-softmax: yields shape (bs, ncat). hard flag controls straight-through discrete sampling
            cat_samples[int(idx_str)] = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        return cont_residual, cat_logits, cat_samples