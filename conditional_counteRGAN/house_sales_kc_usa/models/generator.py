import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualGenerator(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc_in = nn.Linear(input_dim + z_dim + num_classes, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.res3 = ResidualBlock(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, y_onehot, z):
        x_input = torch.cat([x, y_onehot, z], dim=1)
        h = self.fc_in(x_input)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        delta = self.fc_out(h)
        return delta
