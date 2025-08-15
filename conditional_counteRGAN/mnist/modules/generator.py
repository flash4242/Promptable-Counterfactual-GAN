import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, c):
        label = self.label_embed(c.to(x.device)).view(-1, 1, 28, 28)
        x_cat = torch.cat([x, label], dim=1)
        delta = self.net(x_cat)
        return x + delta, delta
