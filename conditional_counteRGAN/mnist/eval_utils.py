import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def evaluate_classifier(classifier, dataloader, device, save_dir=None, prefix="classifier"):
    classifier.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            preds = logits.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({prefix}) - Acc={acc:.4f}")
        plt.savefig(os.path.join(save_dir, f"{prefix}_confusion_matrix.png"))
        plt.close()

    print(f"{prefix} Test Accuracy: {acc:.4f}")


def evaluate_counterfactuals(generator, classifier, x, y_true, y_target, device):
    """
    x: batch of images (B,1,28,28)
    y_true: true labels (B,)
    y_target: target labels for CFs (B,)
    """
    classifier.eval()
    generator.eval()
    num_classes = classifier(torch.zeros(1,1,28,28).to(device)).shape[1]

    y_target_onehot = F.one_hot(y_target, num_classes=num_classes).float().to(device)

    with torch.no_grad():
        residual = generator(x.to(device), y_target.to(device))
        x_cf = x.to(device) + residual
        logits = classifier(x_cf)
        probs = F.softmax(logits, dim=1)

    preds = logits.argmax(1)
    cfr = (preds == y_target.to(device)).float().mean().item()
    pred_gain = (probs[torch.arange(len(y_target)), y_target] -
                 probs[torch.arange(len(y_true)), y_true]).mean().item()

    actionability = (torch.abs(x_cf - x.to(device)).mean().item())

    return {
        "class_flip_rate": cfr,
        "prediction_gain": pred_gain,
        "actionability": actionability,
    }, x_cf.cpu()


def visualize_counterfactual_grid(generator, classifier, dataset, device,
                                  save_path="cf_grid.png"):
    """
    For each digit 0-9: pick one sample, generate CFs for all other digits.
    Output: 10x10 grid (rows=source, cols=target).
    """
    classifier.eval()
    generator.eval()
    num_classes = 10

    # pick one sample per class
    samples = []
    for digit in range(num_classes):
        for img, label in dataset:
            if label == digit:
                samples.append((img.unsqueeze(0), label))
                break

    fig, axes = plt.subplots(num_classes, num_classes, figsize=(15, 15))
    for src_digit in range(num_classes):
        x_src, _ = samples[src_digit]
        x_src = x_src.to(device)
        for tgt_digit in range(num_classes):
            ax = axes[src_digit, tgt_digit]
            ax.axis("off")

            if src_digit == tgt_digit:
                # original image
                ax.imshow(x_src.squeeze().cpu().numpy(), cmap="gray")
            else:
                tgt = torch.tensor([tgt_digit], device=device)
                tgt_onehot = F.one_hot(tgt, num_classes=num_classes).float()

                with torch.no_grad():
                    residual = generator(x_src, tgt)
                    x_cf = x_src + residual

                ax.imshow(x_cf.squeeze().cpu().numpy(), cmap="gray")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved CF grid: {save_path}")
