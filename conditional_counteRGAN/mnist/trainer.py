import torch
import torch.nn as nn
import torch.optim as optim
import os, random
import matplotlib.pyplot as plt
import torch.nn.functional as F

def train_classifier(classifier, train_loader, valid_loader, cfg, device):
    optimizer = optim.Adam(classifier.parameters(), lr=cfg.cls_lr)
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
    opt_g = optim.Adam(generator.parameters(), lr=cfg.g_lr)
    opt_d = optim.Adam(discriminator.parameters(), lr=cfg.d_lr)
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()

    g_losses, d_losses, g_cls_losses = [], [], []

    for epoch in range(cfg.num_epochs_gan):
        g_epoch, d_epoch = 0.0, 0.0
        sum_g_cls = 0.0
        num_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)
            num_batches += 1

            target_y = torch.randint(0, cfg.num_classes, (bs,), device=device)
            target_y = torch.where(target_y == y, (target_y + 1) % cfg.num_classes, target_y)
            
            residual = generator(x, target_y)
            x_cf = torch.clamp(x + residual, -1.0, 1.0)

            # Discriminator update
            opt_d.zero_grad()
            d_real_logits = discriminator(x, y)
            d_fake_logits = discriminator(x_cf.detach(), target_y)
            d_loss = bce(d_real_logits, torch.ones_like(d_real_logits)) \
                   + bce(d_fake_logits, torch.zeros_like(d_fake_logits))
            d_loss.backward()
            opt_d.step()

            # Generator update
            opt_g.zero_grad()
            g_fake_logits = discriminator(x_cf, target_y)
            g_adv = bce(g_fake_logits, torch.ones_like(g_fake_logits))
            g_cls = ce(classifier(x_cf), target_y)
            reg_l1 = torch.abs(residual).mean()

            g_loss = g_adv + cfg.lambda_cls * g_cls + cfg.lambda_reg * reg_l1
            g_loss.backward()
            opt_g.step()

            g_epoch += g_loss.item()
            d_epoch += d_loss.item()
            sum_g_cls += g_cls.item()

        g_losses.append(g_epoch / num_batches)
        d_losses.append(d_epoch / num_batches)
        g_cls_losses.append(sum_g_cls / num_batches)

        print(f"[GAN] Epoch {epoch+1}/{cfg.num_epochs_gan} | "
              f"G: {g_losses[-1]:.4f}, D: {d_losses[-1]:.4f}, "
              f"G_cls: {g_cls_losses[-1]:.4f}")

    save_path = os.path.join(cfg.save_dir, "gan_losses.png")
    plt.figure(figsize=(8, 6))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_cls_losses, label="Classifier Loss (g_cls)", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("CounterGAN Losses")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved GAN loss curves to {save_path}")
