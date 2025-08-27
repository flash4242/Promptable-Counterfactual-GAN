import torch.nn as nn
import torch.nn.functional as F

class NNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super(NNClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(64, output_dim)  # Output layer, no activation
        )

    def forward(self, x):
        return self.net(x)

