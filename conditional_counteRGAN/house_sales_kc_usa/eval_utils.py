import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

def compute_metrics_per_target(generator, classifier, X, y, config, max_vis=500):
    device = config['cuda']
    z_dim = config['z_dim']
    batch_size = config['batch_size']

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_classes = int(np.unique(y_t).size)
    results = []
    originals_vis = []
    cfs_vis = []

    generator.eval()
    classifier.eval()
    with torch.no_grad():
        for target in range(num_classes):
            flips_per_batch = [] # class flip rate
            prox_per_batch = [] # proximity to original
            div_per_batch = [] # diversity of generated samples

            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                mask = (y_batch != target)
                if mask.sum() == 0:
                    continue
                x = x_batch[mask]
                if x.size(0) == 0:
                    continue

                bs = x.size(0)
                target_vec = torch.full((bs,), target, device=device, dtype=torch.long)
                target_onehot = F.one_hot(target_vec, num_classes).float()

                # two noise draws for diversity measurement
                z1 = torch.randn(bs, z_dim, device=device)
                z2 = torch.randn(bs, z_dim, device=device)

                delta1 = generator(x, target_onehot, z1)
                delta2 = generator(x, target_onehot, z2)
                cf1 = x + delta1
                cf2 = x + delta2

                preds = classifier(cf1).argmax(dim=1)
                flip_rate = (preds == target_vec).float().mean().item()
                proximity = torch.mean(torch.abs(delta1)).item()
                diversity = torch.mean(torch.abs(cf1 - cf2)).item()

                flips_per_batch.append(flip_rate)
                prox_per_batch.append(proximity)
                div_per_batch.append(diversity)

                # collect a few for visualization
                if len(originals_vis) < max_vis:
                    originals_vis.append(x.cpu())
                    cfs_vis.append(cf1.cpu())

            # aggregate target
            results.append({
                'target_class': target,
                'class_flip': float(np.mean(flips_per_batch)) if flips_per_batch else np.nan,
                'avg_proximity': float(np.mean(prox_per_batch)) if prox_per_batch else np.nan,
                'diversity': float(np.mean(div_per_batch)) if div_per_batch else np.nan
            })

    # concatenate visuals (limited)
    if originals_vis:
        originals_vis = torch.cat(originals_vis, dim=0).numpy()
        cfs_vis = torch.cat(cfs_vis, dim=0).numpy()
    else:
        originals_vis = np.empty((0, X.shape[1]))
        cfs_vis = np.empty((0, X.shape[1]))

    return pd.DataFrame(results), originals_vis, cfs_vis

def save_metrics(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved metrics to {save_path}")


def plot_tsne(origins, cfs, save_path, n_samples=500):
    if origins.shape[0] == 0:
        print("No samples to plot t-SNE.")
        return
    n = min(origins.shape[0], n_samples)
    data = np.vstack([origins[:n], cfs[:n]])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb = tsne.fit_transform(data)
    plt.figure(figsize=(8,6))
    half = n
    plt.scatter(emb[:half,0], emb[:half,1], s=8, alpha=0.6, label='Originals')
    plt.scatter(emb[half:,0], emb[half:,1], s=8, alpha=0.6, label='Counterfactuals')
    plt.legend()
    plt.title('t-SNE of Originals vs Counterfactuals (sampled)')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE to {save_path}")

# short orchestrator
def evaluate_pipeline(generator, classifier, X_test, y_test, config):
    out = config['out_dir']
    metrics_df, origins_vis, cfs_vis = compute_metrics_per_target(generator, classifier, X_test, y_test, config)
    save_metrics(metrics_df, os.path.join(out, "countergan_metrics.csv"))
    plot_tsne(origins_vis, cfs_vis, os.path.join(out, "tsne_orig_vs_cf.png"))
    return metrics_df
