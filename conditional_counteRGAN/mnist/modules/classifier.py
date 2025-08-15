import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def save_classifier(model, path):
    torch.save(model.state_dict(), path)

def load_classifier(path, device):
    model = Classifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
