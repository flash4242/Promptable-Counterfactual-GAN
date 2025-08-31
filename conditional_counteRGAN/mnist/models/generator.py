import torch
import torch.nn as nn

class ResidualGenerator(nn.Module):
    def __init__(self, img_shape=(1,28,28), num_classes=10):
        super().__init__()
        C,H,W = img_shape
        self.embed = nn.Embedding(num_classes, H*W)

        self.encoder = nn.Sequential(
            nn.Conv2d(C+1, 64, 4, 2, 1),   # 28→14
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),   # 14→7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7→14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 14→28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, target):
        B,C,H,W = x.shape
        y_map = self.embed(target).view(B,1,H,W)
        inp = torch.cat([x, y_map], dim=1)
        h = self.encoder(inp)
        residual = self.decoder(h)
        return residual * 0.8  # scale residual for stability
