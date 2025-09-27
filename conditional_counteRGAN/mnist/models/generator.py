import torch
import torch.nn as nn
from config import Config

class ResidualGenerator(nn.Module):
    def __init__(self, img_shape=(1,28,28), num_classes=Config.num_classes):
        super().__init__()
        C, H, W = img_shape
        self.embed = nn.Embedding(num_classes, H*W)
        self.g_hidden = 32
        self.img_channel = 1

        self.main = nn.Sequential(
            nn.Conv2d(C+1, self.g_hidden, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.g_hidden, self.g_hidden, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.g_hidden, self.g_hidden, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.g_hidden, 1, 3, 1, 1)
        )

    def forward(self, x, target):
        B, C, H, W = x.shape
        y_map = self.embed(target).view(B,1,H,W)
        inp = torch.cat([x, y_map], dim=1)
        residual = self.main(inp)
        return residual
