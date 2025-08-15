import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim * 4)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.adv_head = spectral_norm(nn.Linear(hidden_dim * 4, 1))
        self.cls_head = spectral_norm(nn.Linear(hidden_dim * 4, num_classes))

    def forward(self, x):
        feat = self.feature_extractor(x)
        adv_out = self.adv_head(feat)
        cls_out = self.cls_head(feat)
        return adv_out, cls_out
