# models/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1,28,28), num_classes=10):
        super().__init__()
        C,H,W = img_shape
        self.cond_embed = nn.Embedding(num_classes, H*W)  # condition map

        self.features = nn.Sequential(
            nn.Conv2d(C+1, 16, 4, stride=2, padding=1),   # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, 4, stride=2, padding=1),   # 7x7
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()
        self.adv_head = nn.Linear(16*7*7, 1)            # real/fake
        self.cls_head = nn.Linear(16*7*7, num_classes)  # predicted class

    def forward(self, x, cond_onehot):
        B,C,H,W = x.shape
        cond_map = self.cond_embed(cond_onehot).view(B,1,H,W)
        z = self.features(torch.cat([x, cond_map], dim=1))
        f = self.flatten(z)
        return self.adv_head(f), self.cls_head(f)
