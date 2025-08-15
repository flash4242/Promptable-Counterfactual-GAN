import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),  # 28x28 → 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 14x14 → 7x7
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # 7x7 → 7x7
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1)
            # Nincs sigmoid a vanishing gradient elkerülése miatt
        )

    def forward(self, x, c):
        label = self.label_embed(c.to(x.device)).view(-1, 1, 28, 28)
        x_cat = torch.cat([x, label], dim=1)
        return self.net(x_cat)
