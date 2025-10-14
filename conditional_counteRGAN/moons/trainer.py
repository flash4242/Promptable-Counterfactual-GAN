import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.nn_classifier import NNClassifier
from torch.utils.data import TensorDataset, DataLoader
from models.generator import ResidualGenerator
from models.discriminator import Discriminator

def train_classifier(X_train, y_train, config):
    device = config['cuda']
    clf = NNClassifier(config['input_dim']).to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)

    for _ in range(1000):
        preds = clf(X_t)
        loss = loss_fn(preds, y_t)
        opt.zero_grad(); loss.backward(); opt.step()

    os.makedirs(config['out_dir'], exist_ok=True)
    torch.save({"model_state_dict": clf.state_dict()}, config['clf_model_path'])
    return clf

def train_countergan(config, X_train, y_train, clf_model):
    device = config['cuda']
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = int(np.unique(y_train).size)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    G = ResidualGenerator(config['input_dim'], config['hidden_dim'], num_classes).to(device)
    D = Discriminator(config['input_dim'], config['hidden_dim'], num_classes).to(device)

    opt_G = optim.Adam(G.parameters(), lr=config['lr_G'])
    opt_D = optim.Adam(D.parameters(), lr=config['lr_D'])

    clf_model = clf_model.to(device)
    clf_model.eval()

    d_losses, g_losses = [], []
    batch_inspected = 0

    for epoch in range(config['epochs']):
        epoch_d_losses, epoch_g_losses = [], []
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            bs = x_batch.size(0)
            num_features = x_batch.size(1)

            # target class (ensure different from original)
            target_y = torch.randint(0, num_classes, (bs,), device=device)
            target_y = torch.where(target_y == y_batch, (target_y + 1) % num_classes, target_y)
            target_onehot = F.one_hot(target_y, num_classes).float().to(device)

            # sample binary mask correctly: 0 or 1 per feature
            modifiable_features = torch.randint(0, 2, (bs, num_features), device=device).float()
            raw_residual, masked_residual = G(x_batch, target_onehot, mask=modifiable_features)
            mask_penalty_pre = torch.mean(torch.abs(raw_residual * (1.0 - modifiable_features)))

            x_cf = x_batch + masked_residual

            # D update
            D_real = D(x_batch, F.one_hot(y_batch, num_classes).float())
            D_fake = D(x_cf.detach(), target_onehot)

            D_loss = -D_real.mean() + D_fake.mean()
            opt_D.zero_grad(); D_loss.backward(); opt_D.step()

            # G update
            D_fake_forG = D(x_cf, target_onehot)
            G_adv_loss = -D_fake_forG.mean()
            clf_preds = clf_model(x_cf)
            G_cls_loss = F.cross_entropy(clf_preds, target_y)

            G_reg_loss_l1 = torch.mean(torch.norm(masked_residual, p=1, dim=1))
            G_reg_loss_l2 = torch.mean(torch.norm(masked_residual, p=2, dim=1))

            G_loss = (
                G_adv_loss +
                config['lambda_cls'] * G_cls_loss + 
                config['lambda_reg_l1'] * G_reg_loss_l1 +
                config['lambda_reg_l2'] * G_reg_loss_l2 +
                config['lambda_mask'] * mask_penalty_pre

            )
            opt_G.zero_grad(); G_loss.backward(); opt_G.step()

            epoch_d_losses.append(D_loss.item())
            epoch_g_losses.append(G_loss.item())

            # ---- Logging / diagnostics ----
            with torch.no_grad():
                d_real_p = torch.sigmoid(D_real).mean().item()
                d_fake_p = torch.sigmoid(D_fake.detach()).mean().item()

            if (epoch+1) % (config['epochs']*0.1) == 0 and batch_idx % 5 == 0:
                print(f"[Epoch {epoch+1}/{config['epochs']}] batch {batch_idx} :: "
                    f"D(real)={d_real_p:.3f}, D(fake)={d_fake_p:.3f}, "
                    f"g_adv={G_adv_loss.item():.4f}, g_cls={G_cls_loss.item():.4f}, "
                    f"reg_l1={G_reg_loss_l1.item():.5f}, reg_l2= {G_reg_loss_l2.item():.5f}, mask_pen={mask_penalty_pre.item():.5f}")

        d_losses.append(np.mean(epoch_d_losses))
        g_losses.append(np.mean(epoch_g_losses))

        if (epoch+1) % (config['epochs']*0.2) == 0:
            print(f"[{epoch+1}/{config['epochs']}] D: {d_losses[-1]:.4f}, G: {g_losses[-1]:.4f}")

    os.makedirs(config['out_dir'], exist_ok=True)
    plt.plot(d_losses, label="D_loss")
    plt.plot(g_losses, label="G_loss")
    plt.legend(); plt.savefig(os.path.join(config['out_dir'], "loss_curves.png"))
    plt.close()
    return G