import torch
import torch.nn as nn
from config import Config

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1,28,28), num_classes=Config.num_classes):
        super().__init__()
        C,H,W = img_shape
        self.cond_embed = nn.Embedding(num_classes, H*W)
        self.img_channel = 2
        self.d_hidden = 16

        self.main = nn.Sequential(
            nn.Conv2d(self.img_channel, self.d_hidden, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.d_hidden, self.d_hidden * 2, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.d_hidden * 2, self.d_hidden * 4, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.d_hidden * 4, self.d_hidden * 8, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.flatten = nn.Flatten()
        self.adv_head = nn.Linear(self.d_hidden*8, 1)
        # self.cls_head = nn.Linear(self.d_hidden*8, num_classes)

    def forward(self, x, cond_idx):
        B,C,H,W = x.shape
        cond_map = self.cond_embed(cond_idx).view(B,1,H,W)
        z = self.main(torch.cat([x, cond_map], dim=1))
        f = self.flatten(z)   # [B, 512]
        return self.adv_head(f) #, self.cls_head(f)
