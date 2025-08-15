import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from models.generator import ResidualGenerator
from models.discriminator import Discriminator

def train_countergan(config, X_train, y_train, clf_model):
    device = config['cuda']
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # number of price classes
    num_classes = int(np.unique(y_train).size)

    # DataLoader setup
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # Models
    G = ResidualGenerator(config['input_dim'], config['z_dim'], config['hidden_dim'], num_classes).to(device)
    D = Discriminator(config['input_dim'], config['hidden_dim'], num_classes).to(device)

    opt_G = optim.Adam(G.parameters(), lr=config['lr_G'])
    opt_D = optim.Adam(D.parameters(), lr=config['lr_D'])

    clf_model = clf_model.to(device)
    clf_model.eval()

    d_losses = []
    g_losses = []
    div_losses = []
    for epoch in range(config['epochs']):
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_div_losses = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            bs = x_batch.size(0)

            # sample random target classes different from original
            target_y = torch.randint(0, num_classes, (bs,), device=device)
            target_y = torch.where(target_y == y_batch, (target_y + 1) % num_classes, target_y)
            target_onehot = F.one_hot(target_y, num_classes).float()

            # two noise draws for diversity loss / mode-seeking
            z1 = torch.randn(bs, config['z_dim'], device=device)
            z2 = torch.randn(bs, config['z_dim'], device=device)

            # compute two deltas (residuals)
            delta1 = G(x_batch, target_onehot, z1)  # used for D and for main G loss
            delta2 = G(x_batch, target_onehot, z2)  # used only for diversity loss
            x_cf1 = x_batch + delta1

            # -------------------------
            # Train Discriminator
            # -------------------------
            D_real, D_real_cls = D(x_batch)
            D_fake, _ = D(x_cf1.detach())

            # WGAN-style adversarial loss for D (critic)
            adv_loss_D = -D_real.mean() + D_fake.mean()
            cls_loss_D = F.cross_entropy(D_real_cls, y_batch)
            D_loss = adv_loss_D + config['lambda_cls'] * cls_loss_D

            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            # -------------------------
            # Train Generator
            # -------------------------
            D_fake_forG, _ = D(x_cf1)
            G_adv_loss = -D_fake_forG.mean()

            # classifier (external) encourages x_cf to be classified as target_y
            clf_preds = clf_model(x_cf1)
            G_cls_loss = F.cross_entropy(clf_preds, target_y)

            # regularizer: L1 norm of the delta (proximity)
            G_reg_loss = torch.mean(torch.norm(delta1, p=1, dim=1))

            # diversity (mode-seeking) loss: encourage delta1 != delta2 for different z
            # normalize by latent difference to avoid scaling issues
            numerator = torch.mean(torch.abs(delta1 - delta2))
            denom = torch.mean(torch.abs(z1 - z2)) + config['eps']  # add small value to avoid division by zero
            div_loss = - (numerator / denom)  # negative because we reward maximizing diversity
            # if lambda_div == 0 this term doesn't affect anything

            G_loss = (
                G_adv_loss +
                config['lambda_cls'] * G_cls_loss +
                config['lambda_reg'] * G_reg_loss +
                config['lambda_div'] * div_loss
            )

            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            epoch_d_losses.append(D_loss.item())
            epoch_g_losses.append(G_loss.item())
            epoch_div_losses.append(div_loss.item())

        # epoch log
        d_losses.append(np.mean(epoch_d_losses))
        g_losses.append(np.mean(epoch_g_losses))
        div_losses.append(np.mean(epoch_div_losses))

        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"[{epoch+1}/{config['epochs']}] "
                  f"D: {d_losses[-1]:.4f}, G: {g_losses[-1]:.4f}, div: {div_losses[-1]:.4f}")

    # Save loss curves
    os.makedirs(config['out_dir'], exist_ok=True)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(div_losses, label='Diversity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('CounterGAN Training Loss')
    plt.savefig(os.path.join(config['out_dir'], 'loss_curves.png'))
    plt.close()

    return G
