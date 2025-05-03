#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# ==== Hiperparaméterek ====
params = {
    'batch_size': 128,
    'epochs': 5000,
    'lr_G': 1e-3,
    'lr_D': 1e-3,
    'lambda_cls': 2.0,
    'lambda_reg': 0.5,
    'z_dim': 2,
    'hidden_dim': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==== Adatok előkészítése ====
X, y = make_moons(n_samples=1000, noise=0.1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# ==== Generator ====
class Generator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        delta = self.net(x)
        return x + delta  # Residual output to implement the RGAN idea

# ==== Discriminator ====
class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==== Classifier ====
class Classifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==== Inicializálás ====
G = Generator(hidden_dim=params['hidden_dim']).to(params['device'])
D = Discriminator(hidden_dim=params['hidden_dim']).to(params['device'])
C = Classifier(hidden_dim=params['hidden_dim']).to(params['device'])

# ==== Classifier tanítása ====
clf_optimizer = optim.Adam(C.parameters(), lr=1e-2)
clf_loss_fn = nn.BCELoss()
for _ in range(1000):
    pred = C(train_X.to(params['device']))
    loss = clf_loss_fn(pred, train_y.to(params['device']))
    clf_optimizer.zero_grad()
    loss.backward()
    clf_optimizer.step()

# ==== GAN tanítása ====
optimizer_G = optim.Adam(G.parameters(), lr=params['lr_G'])
optimizer_D = optim.Adam(D.parameters(), lr=params['lr_D'])
bce_loss = nn.BCELoss()

# Freeze classifier weights
for param in C.parameters():
    param.requires_grad = False

for epoch in range(params['epochs']):
    idx = torch.randint(0, train_X.shape[0], (params['batch_size'],))
    x = train_X[idx].to(params['device'])
    y_true = train_y[idx].to(params['device'])

    # === Discriminator ===
    x_cf = G(x)
    d_real = D(x)
    d_fake = D(x_cf.detach())
    loss_D = -torch.mean(torch.log(d_real + 1e-6) + torch.log(1 - d_fake + 1e-6))
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    # === Generator ===
    x_cf = G(x)
    d_fake = D(x_cf)
    y_pred = C(x_cf)

    target_label = 1 - y_true  # counterfactual label
    loss_adv = -torch.mean(torch.log(d_fake + 1e-6))
    loss_cls = bce_loss(y_pred, target_label)
    loss_reg = torch.mean(torch.norm(x_cf - x, p=1, dim=1))

    loss_G = loss_adv + params['lambda_cls'] * loss_cls + params['lambda_reg'] * loss_reg # loss function from the paper

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")

# ==== Counterfactual generálás és megjelenítés ====
G.eval()
with torch.no_grad():
    x_orig = test_X.to(params['device'])
    x_cf = G(x_orig).cpu()


# Teljes adathalmaz kirajzolása háttérként
X_np = X.cpu().numpy()
y_np = y.cpu().numpy().flatten()
plt.figure(figsize=(8, 6))
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='coolwarm', alpha=0.2, label='Teljes adat (háttér)')

# Csak a tesztpontokat és cf-eket külön jelölve
x_orig_np = x_orig.cpu().numpy()
plt.scatter(x_orig_np[:20, 0], x_orig_np[:20, 1], label="Eredeti (teszt)", alpha=0.9, edgecolor='k')
plt.scatter(x_cf[:20, 0], x_cf[:20, 1], label="Ellenpélda", alpha=0.9, marker='x')

# Összekötő vonalak
for i in range(20): # összes kirajzolásához: len(x_orig_np)
    plt.plot([x_orig_np[i, 0], x_cf[i, 0]], [x_orig_np[i, 1], x_cf[i, 1]], 'gray', linestyle='--', alpha=0.5)


plt.legend()
plt.title("counteRGAN: Teljes adathalmaz + Tesztellenpéldák")
plt.grid(True)
plt.savefig("countergan_moons.png")

