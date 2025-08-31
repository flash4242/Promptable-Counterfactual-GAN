import torch
import torch.nn as nn
import torch.optim as optim
import os, random
import matplotlib.pyplot as plt
import torch.nn.functional as F

def train_classifier(classifier, train_loader, valid_loader, cfg, device):
    optimizer = optim.Adam(classifier.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    save_path = cfg.classifier_path

    for epoch in range(cfg.num_epochs_clf):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(x), y)
            loss.backward()
            optimizer.step()

        # validation
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                preds = classifier(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total

        print(f"[Classifier] Epoch {epoch+1}/{cfg.num_epochs_clf} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), save_path)

    print(f"Saved best classifier with acc={best_acc:.4f} to {save_path}")


def train_countergan(generator, discriminator, classifier, train_loader, cfg, device):
    opt_g = optim.Adam(generator.parameters(), lr=cfg.lr)
    opt_d = optim.Adam(discriminator.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()

    g_losses, d_losses = [], []

    for epoch in range(cfg.num_epochs_gan):
        g_epoch, d_epoch = 0.0, 0.0

        # accumulators for detailed components
        sum_d_real_adv = 0.0
        sum_d_fake_adv = 0.0
        sum_d_real_aux = 0.0
        sum_d_fake_aux = 0.0
        sum_g_adv = 0.0
        sum_g_cls = 0.0
        sum_reg_l1 = 0.0

        num_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)
            num_batches += 1

            # ----- sample targets -----
            target_y = torch.randint(0, cfg.num_classes, (bs,), device=device)
            target_y = torch.where(target_y == y, (target_y + 1) % cfg.num_classes, target_y)

            y_onehot      = F.one_hot(y, cfg.num_classes).float()
            target_onehot = F.one_hot(target_y, cfg.num_classes).float()

            residual = generator(x, target_y)
            x_cf = torch.clamp(x + residual, -1.0, 1.0)

            # =======================
            # Discriminator update
            # =======================
            opt_d.zero_grad()

            d_real_logits, d_real_cls = discriminator(x, y)  # Real conditioned on TRUE class
            d_fake_logits, d_fake_cls = discriminator(x_cf.detach(), target_y)  # Fake conditioned on TARGET class

            d_real_adv = bce(d_real_logits, torch.ones_like(d_real_logits))
            d_fake_adv = bce(d_fake_logits, torch.zeros_like(d_fake_logits))

            d_real_aux = ce(d_real_cls, y)  # Aux class: real should be predicted as its true y
            d_fake_aux = ce(d_fake_cls, target_y)  # Aux class: fake should be predicted as target_y

            d_loss = d_real_adv + d_fake_adv + 0.5*(d_real_aux + d_fake_aux)  # 0.5 balances aux
            d_loss.backward()
            opt_d.step()

            # accumulate discriminator components (use .item())
            sum_d_real_adv += d_real_adv.item()
            sum_d_fake_adv += d_fake_adv.item()
            sum_d_real_aux += d_real_aux.item()
            sum_d_fake_aux += d_fake_aux.item()

            # =================
            # Generator update
            # =================
            opt_g.zero_grad()
            g_fake_logits, g_fake_cls = discriminator(x_cf, target_y)
            g_adv = bce(g_fake_logits, torch.ones_like(g_fake_logits))
            g_cls = ce(classifier(x_cf), target_y)           # external classifier guidance

            reg_l1 = torch.abs(residual).mean()

            g_loss = (
                g_adv
              + cfg.lambda_cls  * g_cls          # external classifier agreement
              + cfg.lambda_reg  * reg_l1         # small, targeted edits
            )
            g_loss.backward()
            opt_g.step()

            # accumulate generator components
            sum_g_adv += g_adv.item()
            sum_g_cls += g_cls.item()
            sum_reg_l1 += reg_l1.item()

            g_epoch += g_loss.item()
            d_epoch += d_loss.item()

        # compute per-epoch averages
        avg_d_real_adv = sum_d_real_adv / num_batches
        avg_d_fake_adv = sum_d_fake_adv / num_batches
        avg_d_real_aux = sum_d_real_aux / num_batches
        avg_d_fake_aux = sum_d_fake_aux / num_batches

        avg_g_adv = sum_g_adv / num_batches
        avg_g_cls = sum_g_cls / num_batches
        avg_reg_l1 = sum_reg_l1 / num_batches

        g_losses.append(g_epoch / len(train_loader))
        d_losses.append(d_epoch / len(train_loader))

        print(
            f"[GAN] Epoch {epoch+1}/{cfg.num_epochs_gan} | "
            f"G: {g_losses[-1]:.4f}, D: {d_losses[-1]:.4f} || "
            f"d_real_adv: {avg_d_real_adv:.4f}, d_fake_adv: {avg_d_fake_adv:.4f}, "
            f"d_real_aux: {avg_d_real_aux:.4f}, d_fake_aux: {avg_d_fake_aux:.4f} || "
            f"g_adv: {avg_g_adv:.4f}, g_cls: {avg_g_cls:.4f}, reg_l1: {avg_reg_l1:.6f}"
        )

    # plot and save loss curves
    save_path = os.path.join(cfg.save_dir, "gan_losses.png")
    plt.figure(figsize=(8, 6))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Losses (ACGAN + cycle)")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved GAN loss curves to {save_path}")
