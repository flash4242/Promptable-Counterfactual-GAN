import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), # original img shape (1,28,28), input sahpe (B,1,28,28), output shape (B,32,28,28)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), # output shape (B,64,14,14)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), # output shape (B,128,7,7)
            nn.ReLU(),
            nn.Dropout2d(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
    )


    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
