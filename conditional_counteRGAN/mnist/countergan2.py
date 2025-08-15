#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# CLI argumentumok
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, required=True, help='Cél osztály a counterfactualhoz (0-9)')
parser.add_argument('--data-path', type=str, default='/mnt/data', help='MNIST adat elérési útja')
parser.add_argument('--output-dir', type=str, default='outputs2', help='Kimeneti könyvtár')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ==== Paraméterek ====
params = {
    'batch_size': 128,
    'epochs': 50,
    'lr_G': 1e-3,
    'lr_D': 1e-3,
    'lambda_cls': 3.0,     # Osztály loss hangsúlyosabb, korábbi: 2.0
    'lambda_reg': 0.05,     # Kisebb regularizáció, hogy a delta nagyobb lehessen ha kell, korábbi: 0.1
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==== Adat betöltés ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST betöltése teljes train setként
full_train_dataset = datasets.MNIST(args.data_path, train=True, transform=transform, download=True)

# Train/valid szétválasztás
train_indices, valid_indices = train_test_split(list(range(len(full_train_dataset))), test_size=0.1, stratify=full_train_dataset.targets)
train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
valid_dataset = torch.utils.data.Subset(full_train_dataset, valid_indices)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)

test_dataset = datasets.MNIST(args.data_path, train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ==== Hálózatok ====
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


class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        label = self.label_embed(c.to(x.device)).view(-1, 1, 28, 28)
        x_cat = torch.cat([x, label], dim=1)
        return self.net(x_cat)


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

# ==== Inicializálás ====
G = Generator().to(params['device'])
D = Discriminator().to(params['device'])
C = Classifier().to(params['device'])

# ==== Classifier tanítás ====
clf_optimizer = optim.Adam(C.parameters(), lr=1e-3)
clf_loss_fn = nn.CrossEntropyLoss()
train_losses = []
valid_losses = []

for epoch in range(10):  # növelhető, ha pontosabb classifier kell
    C.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(params['device']), y.to(params['device'])
        logits = C(x)
        loss = clf_loss_fn(logits, y)
        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    C.eval()
    valid_loss = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(params['device']), y.to(params['device'])
            logits = C(x)
            loss = clf_loss_fn(logits, y)
            valid_loss += loss.item() * x.size(0)
    valid_loss /= len(valid_loader.dataset)
    valid_losses.append(valid_loss)

    print(f"Epoch {epoch+1} - Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f}")

# Ábra mentése
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Classifier Train vs Valid Loss')
plt.savefig(os.path.join(args.output_dir, "classifier_loss_plot.png"))
plt.close()


# Fagyasztás
for param in C.parameters():
    param.requires_grad = False

# ==== GAN tanítás ====
optimizer_G = optim.Adam(G.parameters(), lr=params['lr_G'])
optimizer_D = optim.Adam(D.parameters(), lr=params['lr_D'])
bce = nn.BCELoss()
ce = nn.CrossEntropyLoss()

g_losses = []
d_losses = []

for epoch in range(params['epochs']):
    for x, y in train_loader:
        x = x.to(params['device'])
        y_true = y.to(params['device'])
        target = torch.full_like(y_true, args.target)

        # === Discriminator ===
        x_cf, _ = G(x, target)
        d_real = D(x, y_true)
        d_fake = D(x_cf.detach(), target)
        loss_D = -torch.mean(torch.log(d_real + 1e-6) + torch.log(1 - d_fake + 1e-6))
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # === Generator ===
        x_cf, delta = G(x, target)
        d_fake = D(x_cf, target)
        cls_pred = C(x_cf)

        loss_adv = -torch.mean(torch.log(d_fake + 1e-6))
        loss_cls = ce(cls_pred, target)
        loss_reg = torch.mean(torch.abs(delta))

        loss_G = loss_adv + params['lambda_cls'] * loss_cls + params['lambda_reg'] * loss_reg
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{params['epochs']} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")
    g_losses.append(loss_G.item())
    d_losses.append(loss_D.item())


# ==== Counterfactual inferencia & mentés ====
data_iter = iter(test_loader)
imgs, labels = next(data_iter)
imgs = imgs.to(params['device'])
targets = torch.full_like(labels, args.target)

with torch.no_grad():
    cf_imgs, deltas = G(imgs, targets)
    cf_imgs = torch.clamp(cf_imgs, -1, 1)

vutils.save_image(imgs.cpu(), os.path.join(args.output_dir, "original.png"), nrow=4, normalize=True)
vutils.save_image(cf_imgs.cpu(), os.path.join(args.output_dir, "counterfactual.png"), nrow=4, normalize=True)
vutils.save_image((deltas * 0.5 + 0.5).cpu(), os.path.join(args.output_dir, "delta.png"), nrow=4, normalize=True)

# GAN loss ábra mentés
plt.figure()
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Losses')
plt.savefig(os.path.join(args.output_dir, "gan_loss_plot.png"))
plt.close()


print(f"Eredmények mentve az '{args.output_dir}' könyvtárba.")