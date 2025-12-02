import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_classifier(clf, X_test, y_test, config):
    device = config['cuda']
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        preds = clf(X_t).argmax(1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(cm.shape[0])],
                         columns=[f"pred_{i}" for i in range(cm.shape[1])])
    save_path = os.path.join(config["out_dir"], "classifier_confusion.csv")
    os.makedirs(config["out_dir"], exist_ok=True)
    df.to_csv(save_path)
    print(f"Classifier accuracy: {acc:.4f}, confusion matrix saved to {save_path}")


def compute_metrics_per_target(generator, classifier, X, y, config, mask=None):
    """
    mask: None (means generator will be called without mask - fallback),
          or a (input_dim,) or (bs, input_dim) binary float tensor or numpy array.
    """
    device = config['cuda']
    batch_size = config['batch_size']

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_classes = int(np.unique(y_t).size)
    results = []

    generator.eval()
    classifier.eval()
    with torch.no_grad():
        for target in range(num_classes):
            flips_per_batch = []
            pred_gain_per_batch = []
            act_per_batch = []

            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                mask_samples = (y_batch != target)
                if mask_samples.sum() == 0:
                    continue
                x = x_batch[mask_samples]
                if x.size(0) == 0:
                    continue

                bs = x.size(0)
                target_vec = torch.full((bs,), target, device=device, dtype=torch.long)
                target_onehot = F.one_hot(target_vec, num_classes).float().to(device)

                # prepare mask: if mask is None, pass None; if mask is array, expand to bs
                if mask is None:
                    mask_tensor = None
                else:
                    # mask can be numpy 1D or torch 1D or shape (input_dim,)
                    m = torch.tensor(mask, dtype=torch.float32, device=device)
                    if m.dim() == 1:
                        mask_tensor = m.unsqueeze(0).expand(bs, -1)
                    else:
                        mask_tensor = m[:bs].float()  # if provided per-sample
                _, masked_residual = generator(x, target_onehot, mask=mask_tensor)
                cf = x + masked_residual

                cf_logits = classifier(cf)
                cf_preds = cf_logits.argmax(dim=1)
                flip_rate = (cf_preds == target_vec).float().mean().item()

                probs_orig = F.softmax(classifier(x), dim=1)
                probs_cf = F.softmax(cf_logits, dim=1)
                p_orig = probs_orig[torch.arange(bs), target_vec]
                p_cf = probs_cf[torch.arange(bs), target_vec]

                pred_gain = (p_cf - p_orig).mean().item()

                # actionability measured only on masked residual
                actionability = torch.mean(torch.abs(masked_residual)).item()

                flips_per_batch.append(flip_rate)
                pred_gain_per_batch.append(pred_gain)
                act_per_batch.append(actionability)

            results.append({
                "target_class": target,
                "class_flip": float(np.mean(flips_per_batch)) if flips_per_batch else np.nan,
                "prediction_gain": float(np.mean(pred_gain_per_batch)) if pred_gain_per_batch else np.nan,
                "avg_actionability": float(np.mean(act_per_batch)) if act_per_batch else np.nan
            })

    return pd.DataFrame(results)


def plot_decision_boundaries_and_cfs(generator, classifier, X, y, config,
                                     mask=None, n_cf_samples=20, save_prefix="decision_boundaries_cfs"):
    device = config['cuda']
    num_classes = len(np.unique(y))

    # create grid for decision boundaries
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        Z = classifier(grid_t).argmax(1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    # folder ready
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    groups = [
        {"pairs": [(0, 1), (0, 2)], "title": "Counterfactuals from Class 0"},
        {"pairs": [(1, 0), (1, 2)], "title": "Counterfactuals from Class 1"},
        {"pairs": [(2, 0), (2, 1)], "title": "Counterfactuals from Class 2"},
    ]
    colors = ["red", "blue", "green", "orange", "purple", "brown"]

    for g_idx, group in enumerate(groups):
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20, alpha=0.6)

        for (src, tgt) in group["pairs"]:
            pair_idx = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)].index((src, tgt))
            color = colors[pair_idx]

            mask_idx = (y == src)
            if mask_idx.sum() == 0:
                continue
            idx = np.random.choice(np.where(mask_idx)[0], size=min(n_cf_samples, mask_idx.sum()), replace=False)

            x_src = torch.tensor(X[idx], dtype=torch.float32).to(device)
            tgt_vec = torch.full((x_src.size(0),), tgt, device=device, dtype=torch.long)
            tgt_onehot = F.one_hot(tgt_vec, num_classes).float().to(device)

            # prepare mask for these samples
            if mask is None:
                mask_tensor = None
            else:
                m = torch.tensor(mask, dtype=torch.float32, device=device)
                if m.dim() == 1:
                    mask_tensor = m.unsqueeze(0).expand(x_src.size(0), -1)
                else:
                    mask_tensor = m[:x_src.size(0)]
            with torch.no_grad():
                _, masked_residual = generator(x_src, tgt_onehot, mask=mask_tensor)
                x_cf = x_src + masked_residual

            x_src_np = x_src.cpu().numpy()
            x_cf_np = x_cf.cpu().numpy()

            plt.scatter(x_cf_np[:, 0], x_cf_np[:, 1], c=color, label=f"{src}->{tgt}", alpha=0.9)

            # arrows
            for xs, xc in zip(x_src_np, x_cf_np):
                plt.arrow(xs[0], xs[1], xc[0] - xs[0], xc[1] - xs[1],
                          color=color, alpha=0.5, head_width=0.01, length_includes_head=True)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(group["title"])
        plt.legend()
        save_path = os.path.join(out_dir, f"{save_prefix}_group{g_idx+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

def save_metrics(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved metrics to {save_path}")

def plot_decision_boundaries_only(classifier, X, y, config, save_name="decision_boundaries_original.png"):
    device = config['cuda']
    num_classes = len(np.unique(y))

    # Create grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        Z = classifier(grid_t).argmax(1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    # Prepare folder
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=40)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Classifier Decision Boundaries + Original Data Points")

    save_path = os.path.join(out_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def evaluate_pipeline(generator, classifier, X_test, y_test, config):
    # base_out = config['out_dir']

    # # Define masks (as numpy 1D arrays) - input_dim assumed 2
    # masks = {
    #     "both": np.array([1, 1], dtype=np.float32),
    #     "none": np.array([0, 0], dtype=np.float32),
    #     "x_only": np.array([1, 0], dtype=np.float32),
    #     "y_only": np.array([0, 1], dtype=np.float32)
    # }

    # first global classifier metrics
    evaluate_classifier(classifier, X_test, y_test, config)

    all_metrics = {}
    # for name, mask in masks.items():
    #     print(f"Evaluating mask: {name}")
    #     out_dir = os.path.join(base_out, f"mask_{name}")
    #     cfg = dict(config)  # shallow copy
    #     cfg['out_dir'] = out_dir
    #     os.makedirs(out_dir, exist_ok=True)

    #     metrics_df = compute_metrics_per_target(generator, classifier, X_test, y_test, cfg, mask=mask)
    #     save_metrics(metrics_df, os.path.join(out_dir, "metrics.csv"))

    #     plot_decision_boundaries_and_cfs(generator, classifier, X_test, y_test, cfg,
    #                                      mask=mask, n_cf_samples=30,
    #                                      save_prefix=f"decision_boundaries_cfs_{name}")
    #     all_metrics[name] = metrics_df

    # # Optionally concatenate and save a summary table with masks as columns
    # summary_rows = []
    # for name, df in all_metrics.items():
    #     df2 = df.copy()
    #     df2['mask'] = name
    #     summary_rows.append(df2)
    # summary_df = pd.concat(summary_rows, ignore_index=True)
    # save_metrics(summary_df, os.path.join(base_out, "metrics_all_masks.csv"))
    plot_decision_boundaries_only(classifier, X_test, y_test, config,
                              save_name="decision_boundaries_no_cfs.png")

    return None
