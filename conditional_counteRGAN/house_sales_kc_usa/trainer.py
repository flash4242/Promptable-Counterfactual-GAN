import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from models.generator import ResidualGenerator
from models.discriminator import Discriminator
from models.nn_classifier import NNClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn


def train_classifier(X_train_all, X_test, y_train_all, y_test, scaler, config):
    device = config['cuda']
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = os.path.join(config.get('out_dir', '.'), "classifier_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Create a validation split from the training set
    val_frac = config.get('val_frac', 0.10)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=val_frac, random_state=seed, stratify=y_train_all
    )

    num_classes = int(np.unique(y_train_all).size)
    config['num_classes'] = num_classes

    # PyTorch datasets & loaders
    batch_size = config.get('clf_batch_size', config.get('batch_size', 128))
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = NNClassifier(config['input_dim'], output_dim=num_classes).to(device)

    # compute class weights to handle any imbalance
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=config.get('clf_lr', 1e-3), weight_decay=config.get('clf_wd', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_state = None
    patience = config.get('clf_early_stopping', 10)
    wait = 0
    epochs = config.get('clf_epochs', 60)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validation
        model.eval()
        val_running = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_running / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # scheduler and early stopping
        scheduler.step(val_loss)
        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': scaler
            }
            torch.save(best_state, config.get('clf_model_path', os.path.join(out_dir, 'clf_model_best.pth')))
            wait = 0
        else:
            wait += 1

        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | wait={wait}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    # load best model weights
    if best_state is not None:
        model.load_state_dict(best_state['model_state_dict'])

    # Save final model checkpoint (again) with scaler and config
    clf_model_path = config.get('clf_model_path', os.path.join(out_dir, 'clf_model_final.pth'))
    save_dir = os.path.dirname(clf_model_path)

    # only make directory if there is one
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), config['clf_model_path'])
    print(f"Saved classifier model to {clf_model_path}")

    # Plot losses & accuracies
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='train_acc')
    plt.plot(val_accs, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_accuracy_curves.png"), dpi=200)
    plt.close()

    print("Training finished. Best val loss: %.4f" % best_val_loss)

    return model

def grad_norm(params):
    return sum(p.grad.norm().item() for p in params if p.grad is not None)


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
    G = ResidualGenerator(config['input_dim'], config['hidden_dim'], num_classes).to(device)
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
        epoch_pred_gains = []
        epoch_feature_sparsities = [] # proportion of features changed
        epoch_l2_regs = []
        epoch_class_flip_rates = []

        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            bs = x_batch.size(0)
            num_features = x_batch.size(1)

            # sample random target classes different from original
            target_y = torch.randint(0, num_classes, (bs,), device=device)
            target_y = torch.where(target_y == y_batch, (target_y + 1) % num_classes, target_y)
            target_onehot = F.one_hot(target_y, num_classes).float()

            # create random mask only for modifiable features
            modifiable_features = torch.randint(0, 2, (bs, num_features), device=device).float()
            modifiable_features[:, config['immutable_idx']] = 0.0  # force immutable features to stay 0

            raw_residual, masked_residual = G(x_batch, target_onehot, mask=modifiable_features)  # used for D and for main G loss
            x_cf = x_batch + masked_residual

            # penalty for trying to modify immutable features
            mask_penalty_pre = torch.mean(torch.abs(raw_residual * (1 - modifiable_features)))

            # D update
            D_real = D(x_batch, F.one_hot(y_batch, num_classes).float())
            D_fake = D(x_cf.detach(), target_onehot)
            D_loss = -D_real.mean() + D_fake.mean()

            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            # G update
            D_fake_forG = D(x_cf, target_onehot)
            G_adv_loss = -D_fake_forG.mean()

            clf_preds = clf_model(x_cf)  # logits
            G_cls_loss = F.cross_entropy(clf_preds, target_y)

            G_reg_loss = torch.mean(torch.norm(masked_residual, p=1, dim=1))

            G_loss = (
                G_adv_loss +
                G_cls_loss * config['lambda_cls'] +
                G_reg_loss * config['lambda_reg'] +
                mask_penalty_pre * config['lambda_mask']
            )

            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            # ---- Logging / diagnostics ----
            with torch.no_grad():
                d_real_p = torch.sigmoid(D_real).mean().item()
                d_fake_p = torch.sigmoid(D_fake_forG).mean().item()

                # --- Compute classifier outputs ---
                logits_orig = clf_model(x_batch)
                logits_cf = clf_model(x_cf)

                probs_orig = F.softmax(logits_orig, dim=1)
                probs_cf = F.softmax(logits_cf, dim=1)

                # target class probabilities
                p_orig = probs_orig[torch.arange(x_batch.size(0)), target_y]
                p_cf = probs_cf[torch.arange(x_batch.size(0)), target_y]

                # (1) Prediction gain
                pred_gain = (p_cf - p_orig).mean().item()
                epoch_pred_gains.append(pred_gain)

                # (2) Feature modification sparsity
                eps = 1e-3  # small threshold for "change" detection
                feature_changes = (torch.abs(masked_residual) > eps).float()
                sparsity = 1 - feature_changes.mean().item()  # fraction of changed features
                epoch_feature_sparsities.append(sparsity)


                # (3) L2 norm of masked residuals
                reg_loss_l2 = torch.mean(torch.norm(masked_residual, p=2, dim=1))
                epoch_l2_regs.append(reg_loss_l2.item())

                # (4) Class flip rate
                preds_cf = torch.argmax(logits_cf, dim=1)
                class_flip_rate = (preds_cf == target_y).float().mean().item()
                epoch_class_flip_rates.append(class_flip_rate)


            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1}/{config['epochs']}] batch {batch_idx} :: "
                    f"D(real)={d_real_p:.3f}, D(fake)={d_fake_p:.3f}, "
                    f"g_adv={G_adv_loss.item():.4f}, g_cls={G_cls_loss.item():.4f}, "
                    f"reg={G_reg_loss.item():.6f}, mask_pen={mask_penalty_pre.item():.5f}")

            epoch_d_losses.append(D_loss.item())
            epoch_g_losses.append(G_loss.item())

        # epoch log
        d_losses.append(np.mean(epoch_d_losses))
        g_losses.append(np.mean(epoch_g_losses))

        pred_gain_mean = np.mean(epoch_pred_gains)
        sparsity_mean = np.mean(epoch_feature_sparsities)
        l2_reg_mean = np.mean(epoch_l2_regs)
        class_flip_rate_mean = np.mean(epoch_class_flip_rates)

        g_grad_norm = grad_norm(G.parameters())
        d_grad_norm = grad_norm(D.parameters())

        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"[{epoch+1}/{config['epochs']}] "
                  f"D: {d_losses[-1]:.4f}, G: {g_losses[-1]:.4f}, "
                  f"pred_gain={pred_gain_mean:.4f}, sparsity={sparsity_mean:.4f}, "
                  f"l2_reg={l2_reg_mean:.4f}, class_flip_rate={class_flip_rate_mean:.4f}"
                  f" | G_grad: {g_grad_norm:.4f}, D_grad: {d_grad_norm:.4f}")


    # Save loss curves
    os.makedirs(config['out_dir'], exist_ok=True)
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('CounterGAN Training Loss')
    plt.savefig(os.path.join(config['out_dir'], 'loss_curves.png'))
    plt.close()

    return G
