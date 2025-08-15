#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import torch.nn as nn

config = {
    "n_samples": 2000,  # Minták száma az eredeti adathalmazban
    "z_dim": 32,        # Generátor bemeneti dimenziója
    "hidden_dim": 128,  # Rejtett réteg mérete
    "batch_size": 50,   # Batch méret a tanítás során
    "lr": 1e-3,         # Tanulási ráta (learning rate)
    "epochs": 500,        # Tanítási körök száma
    "scale_factor": 10  # A második generált dataset nagysága
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using GPU: {torch.cuda.is_available()}")


def plot_data(ax, X_real, Y_real, X_fake, Y_fake):
    ax.scatter(X_real[:, 0], X_real[:, 1], c='orange', edgecolors='k', label="Original data", marker='o', alpha=0.7)
    ax.scatter(X_fake[:, 0], X_fake[:, 1], c='purple', edgecolors='k', label="Generated data", marker='x', alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    #ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

def build_generator(z_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(z_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2)
    ).to(device)

def build_discriminator(hidden_dim):
    return nn.Sequential(
        nn.Linear(2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid()
    ).to(device)


def train_gan(X, generator, discriminator, config):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["lr"])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])

    loss_D_values, loss_G_values = [], []

    for epoch in range(config["epochs"]):
        np.random.shuffle(X)
        real_samples = torch.from_numpy(X).float()

        loss_D_total, loss_G_total = 0, 0

        for real_batch in real_samples.split(config["batch_size"]):
            # Diszkriminátor tanítása
            z = torch.randn(config["batch_size"], config["z_dim"]).to(device)
            fake_batch = generator(z)

            D_real = discriminator(real_batch.to(device))
            D_fake = discriminator(fake_batch)

            loss_D = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            loss_D_total += loss_D.item()

            # Generátor tanítása
            z = torch.randn(config["batch_size"], config["z_dim"]).to(device)
            fake_batch = generator(z)
            D_fake = discriminator(fake_batch)

            loss_G = -torch.mean(torch.log(D_fake))

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            loss_G_total += loss_G.item()

        loss_D_values.append(loss_D_total)
        loss_G_values.append(loss_G_total)

    return loss_D_values, loss_G_values

def save_loss_plot(loss_D_values, loss_G_values, filename="gan_losses.png"):
    plt.figure()
    plt.plot(range(len(loss_D_values)), loss_D_values, label="Discriminator Loss")
    plt.plot(range(len(loss_G_values)), loss_G_values, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.title("GAN Losses Over Epochs")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def save_generated_data(generator, filename, config, scale_factor=1):
    # Generate fake samples
    z = torch.randn(scale_factor * config["n_samples"], config["z_dim"]).to(device)
    fake_samples = generator(z).cpu().detach().numpy()
    
    # Load or regenerate real data (if needed)
    X_real, Y_real = make_moons(n_samples=config["n_samples"], noise=0.05, random_state=9)
    
    # Assign label 1 to real data, label 0 to fake for plotting (not strictly needed here, just consistent)
    Y_fake = np.full((fake_samples.shape[0],), -1)  # Use -1 to clearly distinguish generated data

    # Plot real and generated data separately
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    plot_data(ax, X_real, Y_real, fake_samples, Y_fake)
    plt.savefig(filename)
    plt.close()


# Moons adathalmaz
X, _ = make_moons(n_samples=config["n_samples"], noise=0.05, random_state=9)

# Modell létrehozása és tanítása
generator = build_generator(config["z_dim"], config["hidden_dim"])
discriminator = build_discriminator(config["hidden_dim"])
loss_D_values, loss_G_values = train_gan(X, generator, discriminator, config)

# plotok mentése
save_loss_plot(loss_D_values, loss_G_values)
save_generated_data(generator, "generated_data.png", config)
#save_generated_data(generator, "generated_data_2.png", config, scale_factor=config["scale_factor"])
