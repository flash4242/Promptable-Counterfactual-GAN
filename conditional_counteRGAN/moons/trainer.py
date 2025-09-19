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
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            bs = x_batch.size(0)
            num_features = x_batch.size(1)

            # target class (ensure different from original)
            target_y = torch.randint(0, num_classes, (bs,), device=device)
            target_y = torch.where(target_y == y_batch, (target_y + 1) % num_classes, target_y)
            target_onehot = F.one_hot(target_y, num_classes).float().to(device)

            # sample binary mask correctly: 0 or 1 per feature
            # Use uniform p=0.5 or make configurable in config
            modifiable_features = torch.randint(0, 2, (bs, num_features), device=device).float()

            residual = G(x_batch, target_onehot, mask=modifiable_features)
            x_cf = x_batch + residual

            # D update
            D_real = D(x_batch)
            D_fake = D(x_cf.detach())
            D_loss = -D_real.mean() + D_fake.mean()
            opt_D.zero_grad(); D_loss.backward(); opt_D.step()

            # G update
            D_fake_forG = D(x_cf)
            G_adv_loss = -D_fake_forG.mean()
            clf_preds = clf_model(x_cf)
            G_cls_loss = F.cross_entropy(clf_preds, target_y)

            # leak_penalty = torch.mean(torch.norm(raw_residual * (1-mask), p=1, dim=1)) # if want to penalize changes on unmodifiable features
            G_reg_loss = torch.mean(torch.norm(residual, p=1, dim=1))

            G_loss = (
                G_adv_loss +
                config['lambda_cls'] * G_cls_loss + 
                config['lambda_reg'] * G_reg_loss
            )
            opt_G.zero_grad(); G_loss.backward(); opt_G.step()

            epoch_d_losses.append(D_loss.item())
            epoch_g_losses.append(G_loss.item())

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
