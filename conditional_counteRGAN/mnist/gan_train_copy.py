#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.utils as vutils

from modules.classifier import Classifier, save_classifier, load_classifier
from modules.generator import Generator
from modules.discriminator import Discriminator

# ==== Paraméterek ====
params = {
    'batch_size': 128,
    'epochs_clf': 10,
    'epochs_gan': 50,
    'lr_G': 5e-5,
    'lr_D': 1e-5,
    'lr_C': 1e-3,
    'lambda_cls': 2.0,
    'lambda_reg': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_dir': 'models',
    'output_dir': 'outputs3'
}

os.makedirs(params['model_dir'], exist_ok=True)
os.makedirs(params['output_dir'], exist_ok=True)

# ==== Adatok betöltése ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train_dataset = datasets.MNIST('/mnt/data', train=True, transform=transform, download=True)
train_indices, valid_indices = train_test_split(list(range(len(full_train_dataset))), test_size=0.1, stratify=full_train_dataset.targets)
train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
valid_dataset = torch.utils.data.Subset(full_train_dataset, valid_indices)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)

# ==== Classifier betöltés vagy tanítás ====
clf_path = os.path.join(params['model_dir'], 'classifier.pt')
if os.path.exists(clf_path):
    print(f"Classifier betöltése: {clf_path}")
    C = load_classifier(clf_path, device=params['device'])
else:
    print("Classifier tanítása...")
    C = Classifier().to(params['device'])
    clf_optimizer = optim.Adam(C.parameters(), lr=params['lr_C'])
    clf_loss_fn = nn.CrossEntropyLoss()
    train_losses, valid_losses = [], []

    for epoch in range(params['epochs_clf']):
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

        print(f"[Classifier] Epoch {epoch+1}/{params['epochs_clf']} - Train: {train_loss:.4f} | Valid: {valid_loss:.4f}")

    save_classifier(C, clf_path)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.legend()
    plt.title('Classifier Training')
    plt.savefig(os.path.join(params['output_dir'], 'classifier_loss.png'))
    plt.close()

# Fagyasztás
for param in C.parameters():
    param.requires_grad = False

# ==== GAN komponensek ====
G = Generator().to(params['device'])
D = Discriminator().to(params['device'])

optimizer_G = optim.Adam(G.parameters(), lr=params['lr_G'])
optimizer_D = optim.Adam(D.parameters(), lr=params['lr_D'])

bce_logits = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()

g_losses, d_losses = [], []

print("GAN tanítása...")
for epoch in range(params['epochs_gan']):
    for x, y in train_loader:
        x = x.to(params['device'])
        y_true = y.to(params['device'])

        random_class = torch.randint(0, 10, (1,), device=params['device']).item()
        target = torch.full_like(y_true, random_class)


        # Discriminator
        x_cf, _ = G(x, target)
        d_real = D(x, y_true)
        d_fake = D(x_cf.detach(), target)
        real_labels = torch.ones_like(d_real)
        fake_labels = torch.zeros_like(d_fake)
        loss_D = bce_logits(d_real, real_labels) + bce_logits(d_fake, fake_labels)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Generator
        x_cf, delta = G(x, target)
        d_fake = D(x_cf, target)
        cls_pred = C(x_cf)

        loss_adv = bce_logits(d_fake, real_labels)
        loss_cls = ce(cls_pred, target)
        loss_reg = torch.mean(torch.abs(delta))

        loss_G = loss_adv + params['lambda_cls'] * loss_cls + params['lambda_reg'] * loss_reg
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"[GAN] Epoch {epoch+1}/{params['epochs_gan']} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")
    d_losses.append(loss_D.item())
    g_losses.append(loss_G.item())

# ==== Teszt és mentés ====
# ==== Evaluation with random targets per image ====
test_loader = DataLoader(
    datasets.MNIST('/mnt/data', train=False, transform=transform),
    batch_size=32, shuffle=True
)
imgs, labels = next(iter(test_loader))
imgs = imgs.to(params['device'])

# Pick random target for each image
targets = torch.randint(0, 10, labels.shape, device=params['device'])

with torch.no_grad():
    cf_imgs, deltas = G(imgs, targets)
    cf_imgs = torch.clamp(cf_imgs, -1, 1)

    # Get classifier predictions
    logits = C(cf_imgs)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    confs = probs[torch.arange(len(preds)), preds]  # confidence values


# Plot originals and counterfactuals with classifier outputs
fig, axes = plt.subplots(2, imgs.size(0), figsize=(2*imgs.size(0), 4))

for i in range(imgs.size(0)):
    # Original image
    axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title(f"Orig: {labels[i].item()}")

    # Counterfactual image with prediction info
    axes[1, i].imshow(cf_imgs[i].cpu().squeeze(), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title(
        f"Tgt: {targets[i].item()}\nPred: {preds[i].item()} ({confs[i]*100:.1f}%)",
        fontsize=8
    )

plt.tight_layout()
plt.savefig(os.path.join(params['output_dir'], "eval_examples_with_preds.png"))
plt.close()



plt.figure()
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.legend()
plt.title('GAN Losses')
plt.savefig(os.path.join(params['output_dir'], 'gan_loss.png'))
plt.close()

print(f"Eredmények mentve a '{params['output_dir']}' mappába.")
