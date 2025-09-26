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

import torch
import matplotlib.pyplot as plt
import numpy as np
def visualize_counterfactual_grid(generator, classifier, dataset, device,
                                  class_map=None, save_path="cf_grid.png",
                                  pick_best_source=False):
    """
    Visualize counterfactuals for all classes.
    - Rows = source class
    - Columns = target class
    - Each cell shows:
        Original (col 0) or Counterfactual (col >0)
        Target class, Predicted class, Prediction confidence
    """
    generator.eval()
    classifier.eval()

    # infer num_classes dynamically
    with torch.no_grad():
        num_classes = classifier(torch.zeros(1, 1, 28, 28).to(device)).shape[1]

    # default mapping: numbers themselves
    if class_map is None:
        class_map = {i: str(i) for i in range(num_classes)}

    # pick one source sample per class
    samples = [None] * num_classes
    best_conf = [-1.0] * num_classes
    with torch.no_grad():
        for x, y in dataset:
            idx = int(y)
            if pick_best_source:
                conf = torch.softmax(classifier(x.unsqueeze(0).to(device)), dim=1)[0, idx].item()
                if conf > best_conf[idx]:
                    samples[idx], best_conf[idx] = x.unsqueeze(0), conf
            else:
                if samples[idx] is None:
                    samples[idx] = x.unsqueeze(0)
            if all(s is not None for s in samples):
                break

    fig, axes = plt.subplots(num_classes, num_classes, figsize=(2.5*num_classes, 2.5*num_classes))

    def show_image(ax, img, tgt, pred=None, conf=None, border_color=None):
        """Helper to display image with annotation."""
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if border_color:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)
        text = f"Tgt={tgt}"
        if pred is not None:
            text += f"\nPred={pred}\nConf={conf:.2f}"
        ax.text(0.5, -0.05, text, color="black", fontsize=7,
                ha="center", va="top", transform=ax.transAxes)

    for r, x_src in enumerate(samples):
        x_src = x_src.to(device)

        for c in range(num_classes):
            ax = axes[r, c]

            if r == c:
                # Original image
                img_disp = ((x_src.squeeze().cpu().numpy() + 1.0) / 2.0)
                src_label = class_map[r]
                show_image(ax, img_disp, tgt=src_label,
                           pred=src_label, conf=1.0, border_color="blue")
            else:
                # Counterfactual to target class
                tgt = torch.tensor([c], device=device)
                with torch.no_grad():
                    residual = generator(x_src, tgt)
                    x_cf = torch.clamp(x_src + residual, -1, 1)
                    logits = classifier(x_cf)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = int(probs.argmax(dim=1))
                    pred_conf = probs[0, pred_idx].item()

                img_disp = ((x_cf.squeeze().cpu().numpy() + 1.0) / 2.0)
                tgt_label = class_map[c]
                pred_label = class_map[pred_idx]
                color = "green" if pred_idx == c else "red"
                show_image(ax, img_disp, tgt=tgt_label,
                           pred=pred_label, conf=pred_conf, border_color=color)

            if r == 0:
                ax.set_title(f"Tgt={class_map[c]}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"Src={class_map[r]}", fontsize=10, rotation=90)

    plt.suptitle("Counterfactual Grid", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved CF grid: {save_path}")

