#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import torch.nn as nn
import torch.nn.functional as F

config = {
    "n_samples": 2000,  # Number of samples in original dataset
    "z_dim": 32,        # Generator input dimension
    "hidden_dim": 128,  # Hidden layer size
    "label_dim": 2,     # Number of possible labels (moons dataset: 2 classes)
    "batch_size": 50,   # Batch size for training
    "lr": 1e-3,         # Learning rate
    "epochs": 500,      # Number of training epochs
    "scale_factor": 10  # Scaling factor for additional generated data
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using GPU: {torch.cuda.is_available()}")

def plot_data(ax, X_real, Y_real, X_fake, Y_fake, title="Generated vs Real Data"):
    ax.scatter(X_real[:, 0], X_real[:, 1], c='orange', edgecolors='k', label="Eredeti adatok", marker='o', alpha=0.7)
    ax.scatter(X_fake[:, 0], X_fake[:, 1], c='purple', edgecolors='k', label="Generált adatok", marker='x', alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)


class Generator(nn.Module):
    def __init__(self, z_dim, label_dim, hidden_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output is 2D (x, y)
        )

    def forward(self, z, label_onehot):
        x = torch.cat([z, label_onehot], 1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, label_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label_onehot):
        x = torch.cat([x, label_onehot], 1)
        return self.net(x)

def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes).float()

# Dataset generation
X, Y = make_moons(n_samples=config["n_samples"], noise=0.05, random_state=9)
Y = Y.astype(np.int64)  # Ensure labels are integer type

# Convert dataset to tensors
real_samples = torch.tensor(X, dtype=torch.float32)
real_labels = torch.tensor(Y, dtype=torch.long)


generator = Generator(config["z_dim"], config["label_dim"], config["hidden_dim"]).to(device)
discriminator = Discriminator(config["label_dim"], config["hidden_dim"]).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["lr"])
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])

# Training loop
loss_D_values, loss_G_values = [], []

for epoch in range(config["epochs"]):
    # Shuffle data
    indices = np.random.permutation(config["n_samples"])
    real_samples, real_labels = real_samples[indices], real_labels[indices]

    loss_D_total, loss_G_total = 0, 0

    for real_batch, real_batch_labels in zip(real_samples.split(config["batch_size"]), real_labels.split(config["batch_size"])):
        real_batch, real_batch_labels = real_batch.to(device), real_batch_labels.to(device)

        # Convert labels to one-hot encoding
        real_labels_onehot = one_hot_encode(real_batch_labels, config["label_dim"]).to(device)

        # Generate fake samples
        z = torch.randn(config["batch_size"], config["z_dim"]).to(device)
        fake_labels = torch.randint(0, config["label_dim"], (config["batch_size"],)).to(device)
        fake_labels_onehot = one_hot_encode(fake_labels, config["label_dim"]).to(device)

        fake_samples = generator(z, fake_labels_onehot)

        # Train Discriminator
        D_real = discriminator(real_batch, real_labels_onehot)
        D_fake = discriminator(fake_samples.detach(), fake_labels_onehot)

        loss_D = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        loss_D_total += loss_D.item()

        # Train Generator
        z = torch.randn(config["batch_size"], config["z_dim"]).to(device)
        fake_labels = torch.randint(0, config["label_dim"], (config["batch_size"],)).to(device)
        fake_labels_onehot = one_hot_encode(fake_labels, config["label_dim"]).to(device)

        fake_samples = generator(z, fake_labels_onehot)
        D_fake = discriminator(fake_samples, fake_labels_onehot)

        loss_G = -torch.mean(torch.log(D_fake))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        loss_G_total += loss_G.item()

    loss_D_values.append(loss_D_total)
    loss_G_values.append(loss_G_total)

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{config['epochs']}], Loss D: {loss_D_total:.4f}, Loss G: {loss_G_total:.4f}")

# Plot loss curves
plt.figure(figsize=(8, 5))
plt.plot(loss_D_values, label="Discriminator Loss", color="red")
plt.plot(loss_G_values, label="Generator Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Conditional GAN Losses")
plt.legend()
plt.grid(True)
plt.savefig("cgan_loss.png")
plt.close()

# Generate and save synthetic data
def save_generated_data(generator, filename, config, scale_factor=1):
    z = torch.randn(scale_factor * config["n_samples"], config["z_dim"]).to(device)
    fake_labels = torch.randint(0, config["label_dim"], (scale_factor * config["n_samples"],)).to(device)
    fake_labels_onehot = one_hot_encode(fake_labels, config["label_dim"]).to(device)

    fake_samples = generator(z, fake_labels_onehot).cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    plot_data(ax, X, Y, fake_samples, fake_labels.cpu().numpy(), title="Generated vs Real Data (CGAN)")
    plt.savefig(filename)
    plt.close()


# 🔹 Save generated datasets
save_generated_data(generator, "cgan_generated_1.png", config)
save_generated_data(generator, "cgan_generated_2.png", config, scale_factor=config["scale_factor"])

print("Training completed and generated images saved!")
