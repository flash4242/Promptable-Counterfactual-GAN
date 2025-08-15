#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=0, help='Cél label az ellenpéldához (0 vagy 1). Default: 1.')
args = parser.parse_args()

def one_hot(labels, num_classes=3):
    return torch.nn.functional.one_hot(labels.squeeze().long(), num_classes=num_classes).float()


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

from sklearn.datasets import make_moons, make_classification

# Moons: két osztály
X_moons, y_moons = make_moons(n_samples=800, noise=0.1)

# Téglalap: harmadik osztály (pl. [-2, 2] x [2, 4])
np.random.seed(42)
X_rect = np.random.uniform(low=[-2, 2], high=[2, 4], size=(400, 2))
y_rect = np.full(400, 2)  # címke: 2

# Egyesítés
X = np.vstack([X_moons, X_rect])
y = np.concatenate([y_moons, y_rect])

# Skálázás és torch konverzió
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long).unsqueeze(1)  # többosztályhoz long típus

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# ==== Generator ====
class Generator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, cond_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, c):
        # c: (batch_size, 1)
        x_cat = torch.cat([x, c], dim=1)
        delta = self.net(x_cat)
        return x + delta # Residual output to implement the RGAN idea

# ==== Discriminator ====
class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, cond_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, c):
        x_cat = torch.cat([x, c], dim=1)
        return self.net(x_cat)


# ==== Classifier ====
class Classifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ==== Inicializálás ====
G = Generator(hidden_dim=params['hidden_dim']).to(params['device'])
D = Discriminator(hidden_dim=params['hidden_dim']).to(params['device'])
C = Classifier(hidden_dim=params['hidden_dim']).to(params['device'])

# ==== Classifier tanítása ====
clf_optimizer = optim.Adam(C.parameters(), lr=1e-2)
clf_loss_fn = nn.CrossEntropyLoss()
for _ in range(1000):
    pred = C(train_X.to(params['device']))
    loss = clf_loss_fn(pred, train_y.squeeze().to(params['device']))
    clf_optimizer.zero_grad()
    loss.backward()
    clf_optimizer.step()

# ==== GAN tanítása ====
optimizer_G = optim.Adam(G.parameters(), lr=params['lr_G'])
optimizer_D = optim.Adam(D.parameters(), lr=params['lr_D'])

# Freeze classifier weights
for param in C.parameters():
    param.requires_grad = False

for epoch in range(params['epochs']):
    idx = torch.randint(0, train_X.shape[0], (params['batch_size'],))
    x = train_X[idx].to(params['device'])
    y_true = train_y[idx].to(params['device'])

    target_label = torch.full((params['batch_size'], 1), float(args.target), device=params['device'])
    target_onehot = one_hot(target_label, num_classes=3).to(params['device'])

    # === Discriminator ===
    x_cf = G(x, target_onehot)
    d_real = D(x, one_hot(y_true, num_classes=3).to(params['device']))
    d_fake = D(x_cf.detach(), target_onehot)
    loss_D = -torch.mean(torch.log(torch.sigmoid(d_real) + 1e-6) + torch.log(1 - torch.sigmoid(d_fake) + 1e-6))
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    # === Generator ===
    x_cf = G(x, target_onehot)
    d_fake = D(x_cf.detach(), target_onehot)
    y_pred = C(x_cf)
    loss_adv = -torch.mean(torch.log(torch.sigmoid(d_fake) + 1e-6))
    loss_cls = clf_loss_fn(y_pred, target_label.squeeze().long())
    loss_reg = torch.mean(torch.norm(x_cf - x, p=1, dim=1))

    # Density regularizáció (valódi célosztály példáktól való távolság)
    target_real = train_X[train_y.squeeze() == args.target].to(params['device'])
    if target_real.shape[0] > 0:
        dist = torch.cdist(x_cf, target_real)
        density_loss = torch.mean(torch.min(dist, dim=1).values)
    else:
        density_loss = torch.tensor(0.0, device=params['device'])

    # Összesített veszteség (paper + density)
    loss_G = loss_adv + params['lambda_cls'] * loss_cls + params['lambda_reg'] * loss_reg + 0.5 * density_loss

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")

# ==== Counterfactual generálás és megjelenítés ====
# Csak azokhoz a példákhoz generálunk CF-et, amelyeknek az eredeti címkéje != target
test_mask = test_y.squeeze() != args.target
x_orig = test_X[test_mask].to(params['device'])
true_labels = test_y[test_mask].to(params['device'])

# Ellenpélda célcímkék: mindenhez a target label
target_cf_label = torch.full((x_orig.shape[0], 1), float(args.target), device=params['device'])
target_cf_onehot = one_hot(target_cf_label, num_classes=3).to(params['device'])

# Ellenpéldák generálása
G.eval()
with torch.no_grad():
    x_cf = G(x_orig, target_cf_onehot).cpu()

# Teljes adathalmaz kirajzolása háttérként
X_np = X.cpu().numpy()
y_np = y.cpu().numpy().flatten()
plt.figure(figsize=(8, 6))
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='coolwarm', alpha=0.2, label='Whole dataset (background)')

# Csak a tesztpontokat és cf-eket külön jelölve
x_orig_np = x_orig.cpu().numpy()
plt.scatter(x_orig_np[:40, 0], x_orig_np[:40, 1], label="Original (test)", alpha=0.9, edgecolor='k')
plt.scatter(x_cf[:40, 0], x_cf[:40, 1], label="Counterfactual", alpha=0.9, marker='x')
# print("x_orig: ", x_orig_np[:40])
# print("x_cf: ", x_cf[:40].cpu().numpy())

# Összekötő vonalak
for i in range(20): # összes kirajzolásához: len(x_orig_np)
    plt.plot([x_orig_np[i, 0], x_cf[i, 0]], [x_orig_np[i, 1], x_cf[i, 1]], 'gray', linestyle='--', alpha=0.5)

plt.legend()
#plt.title("counteRGAN: Teljes adathalmaz + Tesztellenpéldák")
plt.grid(True)
plt.savefig(f"cond_countergan_moons{args.target}.png")