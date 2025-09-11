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
    x: batch of images (B,1,28,28) in normalized [-1,1]
    """
    classifier.eval()
    generator.eval()
    num_classes = classifier(torch.zeros(1,1,28,28).to(device)).shape[1]

    with torch.no_grad():
        residual = generator(x.to(device), y_target.to(device))
        # clamp to training range
        x_cf = torch.clamp(x.to(device) + residual, -1.0, 1.0)
        logits = classifier(x_cf)
        probs = F.softmax(logits, dim=1)

    preds = logits.argmax(1)
    cfr = (preds == y_target.to(device)).float().mean().item()
    pred_gain = (probs[torch.arange(len(y_target)), y_target] -
                 probs[torch.arange(len(y_true)), y_true]).mean().item()

    actionability = (torch.abs(x_cf - x.to(device)).mean().item())

    # return denormalized x_cf for visualization: [-1,1] -> [0,1]
    x_cf_vis = ((x_cf + 1.0) / 2.0).detach().cpu()

    return {
        "class_flip_rate": cfr,
        "prediction_gain": pred_gain,
        "actionability": actionability,
    }, x_cf_vis

def visualize_counterfactual_grid(generator, classifier, dataset, device,
                                  save_path="cf_grid.png"):
    """
    Visualize a grid of counterfactuals for classes the classifier supports.
    Works when dataset & models are filtered to a subset of MNIST digits (e.g. 3 classes).
    """
    classifier.eval()
    generator.eval()

    # infer number of classes from classifier output
    with torch.no_grad():
        num_classes = classifier(torch.zeros(1, 1, 28, 28).to(device)).shape[1]

    # pick one sample per class from dataset (dataset may already be filtered & remapped to 0..num_classes-1)
    samples = []
    found = set()
    for img, label in dataset:
        # label can be tensor or int
        lab = int(label) if isinstance(label, (torch.Tensor,)) else int(label)
        if 0 <= lab < num_classes and lab not in found:
            samples.append((img.unsqueeze(0), lab))
            found.add(lab)
        if len(found) >= num_classes:
            break

    # if we didn't find enough samples, raise informative error
    if len(samples) < num_classes:
        raise ValueError(f"Could not find one sample for each of the {num_classes} classes in the provided dataset. Found samples for classes: {sorted(list(found))}")

    # create grid
    fig, axes = plt.subplots(num_classes, num_classes, figsize=(3*num_classes, 3*num_classes))
    for src_idx in range(num_classes):
        x_src, _ = samples[src_idx]
        x_src = x_src.to(device)
        for tgt_idx in range(num_classes):
            ax = axes[src_idx, tgt_idx] if num_classes > 1 else axes
            ax.axis("off")

            if src_idx == tgt_idx:
                # original image: denormalize for display
                img_disp = ((x_src.squeeze().cpu().numpy() + 1.0) / 2.0)
                ax.imshow(img_disp, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                tgt = torch.tensor([tgt_idx], device=device)
                with torch.no_grad():
                    residual = generator(x_src, tgt)
                    x_cf = torch.clamp(x_src + residual, -1.0, 1.0)
                # move to cpu BEFORE numpy conversion
                img_disp = ((x_cf.squeeze().cpu().numpy() + 1.0) / 2.0)
                ax.imshow(img_disp, cmap="gray", vmin=0.0, vmax=1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved CF grid: {save_path}")
