import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import math
from typing import Tuple, Dict


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
    x_vis = ((x + 1.0) / 2.0).detach().cpu()
    x_cf_vis = ((x_cf + 1.0) / 2.0).detach().cpu()

    return {
        "class_flip_rate": cfr,
        "prediction_gain": pred_gain,
        "actionability": actionability,
    }, (x_vis, x_cf_vis)

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
                    _, masked_residual = generator(x_src, tgt, torch.ones_like(x_src, device=device))
                    x_cf = torch.clamp(x_src + masked_residual, -1, 1)
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


def build_patch_mask_for_batch(x: torch.Tensor, patch_size: int = 5, device=None,
                               shared_per_batch: bool = False) -> torch.Tensor:
    """
    x: (B, C, H, W) image batch (or (B, D) flattened) in [-1,1]
    returns mask tensor of same shape as x with 0/1 float values (1 = modifiable)
    If shared_per_batch=True we pick one mask and copy it to all batch entries.
    """
    if device is None:
        device = x.device
    bs = x.shape[0]
    if x.ndim == 4:
        _, C, H, W = x.shape
        mask = torch.zeros_like(x, device=device)
        # build a single random patch-grid mask (coarse) and optionally replicate
        grid_mask = torch.zeros((1, 1, H, W), device=device)
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                val = float(torch.randint(0, 2, (1,), device=device).item())
                grid_mask[:, :, i:i+patch_size, j:j+patch_size] = val
        if shared_per_batch:
            mask = grid_mask.repeat(bs, C, 1, 1)
        else:
            # different mask per datapoint
            # draw bs independent patch-grids
            mask_list = []
            for _ in range(bs):
                gm = torch.zeros((1, 1, H, W), device=device)
                for i in range(0, H, patch_size):
                    for j in range(0, W, patch_size):
                        val = float(torch.randint(0, 2, (1,), device=device).item())
                        gm[:, :, i:i+patch_size, j:j+patch_size] = val
                mask_list.append(gm)
            mask = torch.cat(mask_list, dim=0)  # (B,1,H,W)
            mask = mask.repeat(1, C, 1, 1)      # match channels
    elif x.ndim == 2:
        # flattened case (B, D) - produce 2D logical mask then flatten
        B, D = x.shape
        # assume we can't patch in flattened easily; produce per-feature random mask
        mask = torch.randint(0, 2, (B, D), device=device).float()
    else:
        raise ValueError("Unsupported x ndim: expected 2 or 4.")
    return mask.float()


# ------------------------------------------------------------------
# Helper: compute requested metrics given raw & masked residuals + mask + classifier
# ------------------------------------------------------------------
def compute_masked_metrics(raw_residual: torch.Tensor, masked_residual: torch.Tensor,
                           x: torch.Tensor, x_cf: torch.Tensor,
                           mask: torch.Tensor, classifier: torch.nn.Module,
                           y_true: torch.Tensor, y_target: torch.Tensor, device) -> Dict:
    """
    Returns dictionary with FR_mac, FR_max, Allowed_L1, mask_penalty_pre.
    - FR_mac: mean flip-rate over batch
    - FR_max: max flip indicator (1.0 if any sample flipped successfully, else 0)
    - Allowed_L1: mean |masked_residual| inside allowed mask
    - mask_penalty_pre: mean |raw_residual * (1-mask)| (residual in forbidden region)
    """
    with torch.no_grad():
        # compute classifier predictions on x_cf
        logits_cf = classifier(x_cf)
        preds_cf = logits_cf.argmax(dim=1)

        flips = (preds_cf == y_target.to(device)).float()
        FR_mac = float(flips.mean().item())
        FR_max = float(flips.max().item())

        # Allowed L1 (within modifiable regions)
        # masked_residual already had mask applied in your generator/trainer, but compute robustly:
        allowed_l1_per_sample = torch.mean(torch.abs(raw_residual * mask), dim=[1,2,3]) \
                                 if raw_residual.ndim==4 else torch.mean(torch.abs(raw_residual * mask), dim=1)
        Allowed_L1 = float(allowed_l1_per_sample.mean().item())

        # penalty in forbidden region (what you called mask_penalty_pre)
        forbidden_penalty_per_sample = torch.mean(torch.abs(raw_residual * (1.0 - mask)), dim=[1,2,3]) \
                                       if raw_residual.ndim==4 else torch.mean(torch.abs(raw_residual * (1.0 - mask)), dim=1)
        mask_penalty_pre = float(forbidden_penalty_per_sample.mean().item())

        # also compute actionability (overall L1 change)
        actionability = float(torch.mean(torch.abs(x_cf - x)).item())

    return {
        "Class_flip_rate_mean": FR_mac,
        "Class_flip_rate_max": FR_max,
        "Residual_L1_norm_in_allowed_patches": Allowed_L1,
        "Actionability (overall L1 norm)": actionability,
        "mask_penalty_pre": mask_penalty_pre
    }

def make_and_save_heatmaps(x: torch.Tensor, x_cf: torch.Tensor, mask: torch.Tensor,
                           metrics: Dict, save_dir: str,
                           y_true: torch.Tensor = None, y_target: torch.Tensor = None,
                           max_samples: int = 16):
    """
    Create per-sample visualizations and a tiled grid for the batch.
    For each sample we save:
        [Original | Counterfactual | |Δ| heatmap | Patch-mask overlay]
    Titles show source→target classes and key metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- convert to CPU numpy [B,H,W] ---
    def to_numpy_img(t):
        t = t.detach().cpu()
        if t.ndim == 4 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.ndim == 3:
            return t.numpy()
        raise ValueError("Expected (B,1,H,W) or (B,H,W)")

    x_np = to_numpy_img(x)
    xcf_np = to_numpy_img(x_cf)
    diff_np = np.abs(xcf_np - x_np)
    mask_np = to_numpy_img(mask)

    B = x_np.shape[0]
    n_show = min(B, max_samples)

    # --- per-sample figures ---
    for i in range(n_show):
        fig, axs = plt.subplots(1, 4, figsize=(10, 2.8), constrained_layout=True)

        axs[0].imshow(x_np[i], cmap="gray", vmin=0, vmax=1)
        axs[0].set_title("Original", fontsize=9)
        axs[0].axis("off")

        axs[1].imshow(xcf_np[i], cmap="gray", vmin=0, vmax=1)
        axs[1].set_title("Counterfactual", fontsize=9)
        axs[1].axis("off")

        im = axs[2].imshow(diff_np[i], cmap="hot", vmin=0, vmax=max(1e-6, diff_np.max()))
        axs[2].set_title("|Δ|", fontsize=9)
        axs[2].axis("off")
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.02)

        axs[3].imshow(x_np[i], cmap="gray", vmin=0, vmax=1)
        axs[3].imshow(mask_np[i], cmap="Greens", alpha=0.5, vmin=0, vmax=1)
        axs[3].set_title("Patch mask\n(green = modifiable)", fontsize=8)
        axs[3].axis("off")

        # --- Add a clean super-title with classes + key metrics ---
        src_label = int(y_true[i].item()) if y_true is not None else "?"
        tgt_label = int(y_target[i].item()) if y_target is not None else "?"
        plt.suptitle(
            f"Src={src_label} → Tgt={tgt_label} | "
            f"FR_mac={metrics['Class_flip_rate_mean']:.3f} | "
            f"Allowed_L1={metrics['Residual_L1_norm_in_allowed_patches']:.4f}",
            fontsize=10, y=1.05
        )

        save_p = os.path.join(save_dir, f"sample_{i}_src{src_label}_tgt{tgt_label}.png")
        plt.savefig(save_p, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- batch overview ---
    cols = 4
    rows = math.ceil(n_show / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), constrained_layout=True)
    axs = axs.flatten()
    for idx in range(rows * cols):
        ax = axs[idx]
        if idx < n_show:
            ax.imshow(x_np[idx], cmap="gray", vmin=0, vmax=1)
            ax.imshow(mask_np[idx], cmap="Greens", alpha=0.45, vmin=0, vmax=1)
            src = int(y_true[idx].item()) if y_true is not None else "?"
            tgt = int(y_target[idx].item()) if y_target is not None else "?"
            ax.set_title(f"{src}→{tgt}", fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle(
        f"batch overview | Class_flip_rate_mean={metrics['Class_flip_rate_mean']:.3f} | Residual_L1_norm_in_allowed_patches={metrics['Residual_L1_norm_in_allowed_patches']:.4f}",
        fontsize=10, y=1.02
    )
    batch_save_p = os.path.join(save_dir, f"batch_overview.png")
    plt.savefig(batch_save_p, dpi=180, bbox_inches="tight")
    plt.close(fig)

def visualize_patch_grid(img, patch_size, save_path=None, alpha=0.6):
    """
    Visualize patch grid overlay on top of an MNIST-like image.

    Args:
        img (torch.Tensor or np.ndarray): Image in range [0,1] or [-1,1], shape (1,28,28) or (28,28)
        patch_size (int): Patch size in pixels (e.g., 7 -> 4x4 grid)
        save_path (str): If given, saves figure to this path
        alpha (float): Overlay transparency
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().squeeze().numpy()
    if img.min() < 0:  # convert [-1,1] -> [0,1]
        img = (img + 1) / 2.0

    h, w = img.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    total_patches = num_patches_h * num_patches_w

    fig, ax = plt.subplots(figsize=(4, 4))

    # Proper extent alignment ensures grid lines sit on pixel boundaries
    ax.imshow(img, cmap="Greens", alpha=alpha, extent=[0, w, h, 0])

    # Grid lines exactly on patch borders
    ax.set_xticks(np.arange(0, w + 1, patch_size))
    ax.set_yticks(np.arange(0, h + 1, patch_size))
    ax.grid(color="lime", linewidth=1)

    # Draw patch numbers centered in each patch
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch_idx = i * num_patches_w + j
            cx = j * patch_size + patch_size / 2
            cy = i * patch_size + patch_size / 2
            ax.text(cx, cy, str(patch_idx), ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f"Patch Grid ({num_patches_h}×{num_patches_w}, total={total_patches})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_pipeline(generator, classifier, x, y, full_dataset, config):
    """
    Runs one batch from test_loader, builds random 5x5 patch masks per datapoint,
    computes counterfactuals via the generator (calling the mask-aware forward),
    computes metrics (FR_mac, FR_max, Allowed_L1, mask_penalty_pre),
    and saves visualizations per-sample + batch overview.
    """
    device = config.device
    generator.eval()
    classifier.eval()

    x, y = x.to(device), y.to(device)
    bs = x.size(0)
    y_target = torch.randint(0, config.num_classes, y.shape, device=device)
    mask_eq = (y_target == y)
    y_target[mask_eq] = (y_target[mask_eq] + 1) % config.num_classes # ensure target != source

    # build 5x5 patch masks (different mask per datapoint)
    mask = build_patch_mask_for_batch(x, patch_size=config.patch_size,
                                        device=device, shared_per_batch=False)

    with torch.no_grad():
        raw_residual, masked_residual = generator(x, y_target.to(device), mask.to(device))
        x_cf = torch.clamp(x + masked_residual, -1.0, 1.0)

    metrics = compute_masked_metrics(raw_residual, masked_residual, x, x_cf, mask, classifier, y, y_target, device)

    # prepare visuals: convert to [0,1] for display
    x_vis = ((x + 1.0) / 2.0).detach().cpu()
    x_cf_vis = ((x_cf + 1.0) / 2.0).detach().cpu()
    mask_vis = mask.detach().cpu()

    # save heatmaps per sample and batch overview
    save_dir = os.path.join(config.save_dir, "eval_visuals")
    make_and_save_heatmaps(x_vis, x_cf_vis, mask_vis, metrics, save_dir=save_dir,y_true=y, y_target=y_target, max_samples=min(16, bs))
    visualize_counterfactual_grid(generator, classifier, full_dataset, device, save_path=os.path.join(config.save_dir, "cf_grid.png"))
    visualize_patch_grid(x_vis[0], patch_size=config.patch_size, save_path=os.path.join(save_dir, "patch_grid.png"))


    print("Evaluation metrics:", metrics)

