# eval_utils.py
import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Optional external mask analysis function (user already had)
try:
    from eval_utils_mask_analysis import evaluate_pipeline_masks
except Exception:
    evaluate_pipeline_masks = None  # If not available, evaluator will skip mask analysis


def compute_metrics_per_target(generator: torch.nn.Module,
                               classifier: torch.nn.Module,
                               X: np.ndarray,
                               y: np.ndarray,
                               config: dict,
                               max_vis: int = 500) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    For each target class t compute:
      - class_flip (validity): fraction of samples moved to predicted target class
      - prediction_gain: mean increase in classifier probability for target class
      - avg_actionability: mean(|masked_residual|) averaged across samples
    Also returns a small set of original/cf pairs for visual inspection (limited by max_vis).
    Enforces immutable features if config['immutable_idx'] is provided.
    Assumes X is already scaled (MinMax to [0,1]).
    """
    device = config.get('cuda', 'cpu')
    batch_size = int(config.get('batch_size', 128))

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_classes = int(np.unique(y).size)
    results = []
    originals_vis = []
    cfs_vis = []

    generator.eval()
    classifier.eval()

    immutable_idx = config.get('immutable_idx', [])

    with torch.no_grad():
        for target in range(num_classes):
            flips = []
            pred_gains = []
            actions = []

            for x_batch, y_batch in loader:
                # only consider samples whose original class != target (we want counterfactuals)
                mask_samples = (y_batch != target)
                if mask_samples.sum() == 0:
                    continue

                x = x_batch[mask_samples].to(device)
                y_sel = y_batch[mask_samples].to(device)
                bs = x.size(0)

                # build target vectors
                target_vec = torch.full((bs,), target, dtype=torch.long, device=device)
                target_onehot = F.one_hot(target_vec, num_classes).float().to(device)

                # create mask that allows changes for all features except immutable ones
                mask_tensor = torch.ones_like(x, device=device)
                if immutable_idx:
                    mask_tensor[:, immutable_idx] = 0.0

                # generator output (raw, masked residual)
                _, masked_residual = generator(x, target_onehot, mask=mask_tensor)
                x_cf = (x + masked_residual).detach()

                # classifier outputs
                logits_orig = classifier(x)
                logits_cf = classifier(x_cf)
                probs_orig = F.softmax(logits_orig, dim=1)
                probs_cf = F.softmax(logits_cf, dim=1)

                # flip rate: fraction where predicted class becomes the target
                preds_cf = torch.argmax(logits_cf, dim=1)
                flip_rate = (preds_cf == target_vec).float().mean().item()

                # prediction gain: mean increase in the classifier's probability for the target class
                p_orig = probs_orig[torch.arange(bs), target_vec]
                p_cf = probs_cf[torch.arange(bs), target_vec]
                pred_gain = (p_cf - p_orig).mean().item()

                # actionability: mean absolute modification (normalized units)
                avg_action = torch.mean(torch.abs(masked_residual)).item()

                flips.append(flip_rate)
                pred_gains.append(pred_gain)
                actions.append(avg_action)

                # collect visuals (limited)
                if len(originals_vis) < max_vis:
                    originals_vis.append(x.cpu())
                    cfs_vis.append(x_cf.cpu())

            results.append({
                'target_class': int(target),
                'class_flip': float(np.mean(flips)) if flips else float('nan'),
                'prediction_gain': float(np.mean(pred_gains)) if pred_gains else float('nan'),
                'avg_actionability': float(np.mean(actions)) if actions else float('nan')
            })

    if originals_vis:
        originals_vis = torch.cat(originals_vis, dim=0).numpy()
        cfs_vis = torch.cat(cfs_vis, dim=0).numpy()
    else:
        originals_vis = np.empty((0, X.shape[1]))
        cfs_vis = np.empty((0, X.shape[1]))

    metrics_df = pd.DataFrame(results)
    return metrics_df, originals_vis, cfs_vis


def analyze_feature_shift_importance(X_orig: np.ndarray,
                                     X_cf: np.ndarray,
                                     feature_names: list,
                                     save_path: str,
                                     scaler: Optional[object] = None) -> pd.DataFrame:
    """
    Global feature shift importance:
      - mean |Δ| per feature in normalized [0,1] units (i.e., fraction of the feature range).
      - If scaler (MinMaxScaler) is provided, compute and also include denormalized mean absolute change.
    Produces a horizontal bar chart with percent-of-range labels.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if X_orig.size == 0 or X_cf.size == 0:
        print("[feature_shift] No visuals available, skipping feature shift plot.")
        return pd.DataFrame({'feature': feature_names, 'mean_abs_change_norm': [0.0] * len(feature_names)})

    delta = np.mean(np.abs(X_cf - X_orig), axis=0)  # normalized units (0..1)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_change_norm': delta,
        'mean_pct_of_range': delta * 100.0
    }).sort_values(by='mean_abs_change_norm', ascending=False).reset_index(drop=True)

    # if scaler is provided and has data_min_ / data_max_, compute denorm mean abs change
    if scaler is not None and hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
        try:
            data_range = (scaler.data_max_ - scaler.data_min_).astype(float)
            # mean absolute denormalized change (approx): delta * data_range
            mean_abs_denorm = delta * data_range
            importance_df['mean_abs_change_denorm'] = mean_abs_denorm
        except Exception:
            importance_df['mean_abs_change_denorm'] = np.nan

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df['feature'], importance_df['mean_pct_of_range'], color='cornflowerblue')
    plt.gca().invert_yaxis()
    plt.title('Global Feature Shift Importance (Mean |Δ| per feature)', fontsize=14)
    plt.xlabel('Mean |Δ| per feature — % of full feature range (0..100%)', fontsize=11)

    xmax = max(importance_df['mean_pct_of_range'].max() * 1.1, 1.0)
    plt.xlim(0, xmax)

    # annotate values at the end of bars
    for bar in bars:
        w = bar.get_width()
        plt.text(w + xmax * 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{w:.2f}%", va='center', fontsize=9)

    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[feature_shift] Saved feature-shift importance plot to {save_path}")
    return importance_df


def analyze_class_pair_sensitivity(G: torch.nn.Module,
                                   clf: torch.nn.Module,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: list,
                                   config: dict,
                                   save_dir: str) -> np.ndarray:
    """
    For each source class s and each target class t != s compute mean |Δ| per feature
    (in normalized units). Save one heatmap per source class:
      - rows = features
      - cols = target classes (excluding same-class)
    Heatmap color scale clipped at a high percentile for contrast.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = config.get('cuda', 'cpu')
    num_classes = int(np.unique(y).size)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    G.eval()
    clf.eval()

    immutable_idx = config.get('immutable_idx', [])

    deltas = np.zeros((num_classes, num_classes, X.shape[1]), dtype=float)  # (src, tgt, feature)

    with torch.no_grad():
        for s in range(num_classes):
            x_src = X_t[y_t == s]
            if x_src.size(0) == 0:
                continue

            # prepare a mask that disallows immutable changes
            mask_template = torch.ones((x_src.size(0), X.shape[1]), dtype=torch.float32, device=device)
            if immutable_idx:
                mask_template[:, immutable_idx] = 0.0

            for t in range(num_classes):
                if s == t:
                    continue
                target_vec = torch.full((x_src.size(0),), t, device=device, dtype=torch.long)
                target_onehot = F.one_hot(target_vec, num_classes).float().to(device)

                _, masked_residual = G(x_src, target_onehot, mask=mask_template)
                # mean absolute residual per feature across the source samples
                mean_abs = torch.mean(torch.abs(masked_residual), dim=0).cpu().numpy()
                deltas[s, t, :] = mean_abs

    # Visualize per source class
    for s in range(num_classes):
        valid_targets = [t for t in range(num_classes) if t != s]
        if not valid_targets:
            continue
        delta_matrix = deltas[s, valid_targets, :].T  # shape: (features, n_targets)

        # avoid all-zero matrix coloring issues
        if np.allclose(delta_matrix, 0.0):
            vmax = 1e-3
        else:
            vmax = np.percentile(delta_matrix, 99)

        # dynamic figure sizing: keep heatmap readable even for many features
        width = max(2.0 + 0.8 * len(valid_targets), 4.0)
        height = max(0.28 * len(feature_names), 6.0)
        plt.figure(figsize=(width, height))
        sns.heatmap(delta_matrix,
                    cmap="YlGnBu",
                    xticklabels=[f"→{t}" for t in valid_targets],
                    yticklabels=feature_names,
                    vmin=0.0,
                    vmax=vmax if vmax > 0 else None,
                    cbar_kws={"label": "Mean |Δ| (fraction of feature range [0..1])"},
                    linewidths=0.25)
        plt.title(f"Feature Sensitivity for Source Class {s}", fontsize=13)
        plt.xlabel("Target Class")
        plt.ylabel("Feature")
        plt.tight_layout()
        fname = os.path.join(save_dir, f"class_pair_sensitivity_src{s}.png")
        plt.savefig(fname, dpi=200)
        plt.close()

    print(f"[class_sensitivity] Saved class-pair sensitivity heatmaps to {save_dir}")
    return deltas

def evaluate_classifier(classifier: torch.nn.Module,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        out_dir: Optional[str] = None,
                        device: str = 'cpu',
                        class_names: Optional[list] = None) -> dict:
    """
    Evaluate classifier and save confusion matrix and a textual report.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    classifier = classifier.to(device)
    classifier.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = classifier(X_test_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = np.array(y_test, dtype=int)

    acc = float(accuracy_score(y_true, preds))
    precision = float(precision_score(y_true, preds, average='weighted', zero_division=0))
    recall = float(recall_score(y_true, preds, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, preds, average='weighted', zero_division=0))
    report = classification_report(y_true, preds, target_names=class_names, digits=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Classifier Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # save textual report
    with open(os.path.join(out_dir, "classifier_report.txt"), "w") as f:
        f.write("=== Classifier Evaluation ===\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n\n")
        f.write(report)

    print(f"[classifier_eval] Accuracy: {acc:.4f}, F1: {f1:.4f}. Saved confusion matrix and report to {out_dir}")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report
    }


def generate_case_study_report_from_vis(
        classifier: torch.nn.Module,
        originals: np.ndarray,
        cfs: np.ndarray,
        config: dict,
        scaler: Optional[object] = None,
        n_samples: int = 20,
        eps: float = 1e-3,
        save_dir: Optional[str] = None,
        random_seed: int = 42,
        top_k_features: int = 5
):
    """
    Build case-study CSVs & summary from already-collected originals and cfs.
    - saves per-sample feature CSVs in subfolders case_studies/samples/src{SRC}_tgt{TGT}/
    - returns (df_summary, feature_summary, aggregate)
    Assumes originals and cfs are normalized to [0,1] if scaler is MinMax.
    """
    assert originals is not None and cfs is not None, "originals and cfs required"
    assert originals.shape == cfs.shape, "originals and cfs must have same shape"

    if save_dir is None:
        out = config.get('out_dir', '.')
        save_dir = os.path.join(out, "case_studies")
    os.makedirs(save_dir, exist_ok=True)

    samples_base = os.path.join(save_dir, "samples")
    os.makedirs(samples_base, exist_ok=True)

    feature_names = config.get('feature_names', [f"feat_{i}" for i in range(originals.shape[1])])
    d = originals.shape[1]
    N = originals.shape[0]
    n_samples = min(n_samples, N)
    rng = np.random.default_rng(random_seed)
    idxs = rng.choice(np.arange(N), size=n_samples, replace=False)

    device = config.get('cuda', 'cpu')
    classifier = classifier.to(device)
    classifier.eval()

    # classifier predictions & probabilities for selected samples
    with torch.no_grad():
        orig_t = torch.tensor(originals[idxs], dtype=torch.float32, device=device)
        cf_t = torch.tensor(cfs[idxs], dtype=torch.float32, device=device)
        logits_orig = classifier(orig_t)
        logits_cf = classifier(cf_t)
        probs_orig = F.softmax(logits_orig, dim=1).cpu().numpy()
        probs_cf = F.softmax(logits_cf, dim=1).cpu().numpy()
        preds_orig = np.argmax(probs_orig, axis=1)
        preds_cf = np.argmax(probs_cf, axis=1)

    # intended targets (where classifier gained most)
    delta_probs = probs_cf - probs_orig
    intended_targets = np.argmax(delta_probs, axis=1)

    abs_diff_norm = np.abs(cfs[idxs] - originals[idxs])  # normalized abs change
    num_changed = (abs_diff_norm > eps).sum(axis=1)
    frac_changed = num_changed / float(d)
    sparsity_per_sample = 1.0 - frac_changed
    L1_sum = np.sum(abs_diff_norm, axis=1)
    L1_mean = np.mean(abs_diff_norm, axis=1)
    L2 = np.linalg.norm(abs_diff_norm, axis=1)

    pred_gain = probs_cf[np.arange(n_samples), intended_targets] - probs_orig[np.arange(n_samples), intended_targets]
    success_bool = (preds_cf == intended_targets).astype(int)

    # denormalize if scaler is provided
    orig_denorm = cf_denorm = abs_diff_denorm = feature_range = None
    if scaler is not None:
        try:
            orig_denorm = scaler.inverse_transform(originals[idxs])
            cf_denorm = scaler.inverse_transform(cfs[idxs])
            abs_diff_denorm = np.abs(cf_denorm - orig_denorm)
            if hasattr(scaler, 'data_max_') and hasattr(scaler, 'data_min_'):
                feature_range = (scaler.data_max_ - scaler.data_min_).astype(float)
        except Exception:
            orig_denorm = cf_denorm = abs_diff_denorm = feature_range = None

    summary_rows = []
    for i_local, i_global in enumerate(idxs):
        src_pred = int(preds_orig[i_local])
        intended_t = int(intended_targets[i_local])
        cf_pred = int(preds_cf[i_local])
        sample_dir = os.path.join(samples_base, f"src{src_pred}_tgt{intended_t}")
        os.makedirs(sample_dir, exist_ok=True)

        row = {
            "sample_idx_in_vis": int(i_global),
            "source_pred": int(src_pred),
            "intended_target": int(intended_t),
            "cf_pred": int(cf_pred),
            "success": int(success_bool[i_local]),
            "prediction_gain": float(pred_gain[i_local]),
            "num_changed": int(num_changed[i_local]),
            "frac_changed": float(frac_changed[i_local]),
            "sparsity": float(sparsity_per_sample[i_local]),
            "L1_sum_norm": float(L1_sum[i_local]),
            "L1_mean_norm": float(L1_mean[i_local]),
            "L2_norm": float(L2[i_local])
        }

        topk_idx = np.argsort(-abs_diff_norm[i_local])[:top_k_features]
        topk_feats = [feature_names[j] for j in topk_idx]
        topk_vals_pct = (abs_diff_norm[i_local, topk_idx] * 100.0).round(6).tolist()
        row["topk_features"] = ";".join(topk_feats)
        row["topk_vals_pct_of_range"] = ";".join(map(str, topk_vals_pct))

        # save per-feature CSV for this sample into its src->tgt folder
        df_feat = pd.DataFrame({
            "feature": feature_names,
            "orig_norm": originals[i_global],
            "cf_norm": cfs[i_global],
            "abs_delta_norm": abs_diff_norm[i_local],
            "pct_of_range": (abs_diff_norm[i_local] * 100.0)
        })
        if orig_denorm is not None:
            df_feat["orig_denorm"] = orig_denorm[i_local]
            df_feat["cf_denorm"] = cf_denorm[i_local]
            df_feat["abs_delta_denorm"] = abs_diff_denorm[i_local]
            if feature_range is not None:
                df_feat["pct_of_range_denorm"] = (abs_diff_denorm[i_local] / (feature_range + 1e-12)) * 100.0

        sample_fname = os.path.join(sample_dir, f"sample_{i_global}_features.csv")
        df_feat.to_csv(sample_fname, index=False)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # feature level aggregates across selected samples
    mean_abs_delta_norm = abs_diff_norm.mean(axis=0)
    change_freq = (abs_diff_norm > eps).mean(axis=0)
    mean_pct_of_range = mean_abs_delta_norm * 100.0

    feature_summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_delta_norm": mean_abs_delta_norm,
        "mean_pct_of_range": mean_pct_of_range,
        "change_freq": change_freq
    }).sort_values(by="mean_abs_delta_norm", ascending=False)

    if abs_diff_denorm is not None:
        mean_abs_delta_denorm = abs_diff_denorm.mean(axis=0)
        feature_summary["mean_abs_delta_denorm"] = mean_abs_delta_denorm

    # save summary files
    df_summary.to_csv(os.path.join(save_dir, "case_study_sample_summary.csv"), index=False)
    feature_summary.to_csv(os.path.join(save_dir, "case_study_feature_summary.csv"), index=False)

    aggregate = {
        "n_samples_used": int(n_samples),
        "class_flip_rate": float(df_summary["success"].mean()) if not df_summary.empty else float('nan'),
        "mean_prediction_gain": float(df_summary["prediction_gain"].mean()) if not df_summary.empty else float('nan'),
        "median_prediction_gain": float(df_summary["prediction_gain"].median()) if not df_summary.empty else float('nan'),
        "mean_L1_sum_norm": float(df_summary["L1_sum_norm"].mean()) if not df_summary.empty else float('nan'),
        "mean_L2_norm": float(df_summary["L2_norm"].mean()) if not df_summary.empty else float('nan'),
        "mean_num_changed": float(df_summary["num_changed"].mean()) if not df_summary.empty else float('nan'),
        "mean_frac_changed": float(df_summary["frac_changed"].mean()) if not df_summary.empty else float('nan'),
        "mean_sparsity": float(df_summary["sparsity"].mean()) if not df_summary.empty else float('nan')
    }

    # top-k global features (by mean_abs_delta_norm)
    topk_global_idx = np.argsort(-mean_abs_delta_norm)[:top_k_features]
    aggregate["topk_features_global"] = ",".join([feature_names[j] for j in topk_global_idx])
    aggregate["topk_features_global_mean_pct"] = ",".join([f"{(mean_abs_delta_norm[j]*100.0):.3f}" for j in topk_global_idx])

    pd.DataFrame([aggregate]).to_csv(os.path.join(save_dir, "case_study_aggregate_summary.csv"), index=False)

    print(f"[case_study] Saved {n_samples} per-sample CSVs (grouped by src->tgt), sample_summary, feature_summary and aggregate to {save_dir}")
    return df_summary, feature_summary, aggregate


def save_metrics(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[metrics] Saved metrics to {save_path}")


def evaluate_pipeline(generator: torch.nn.Module,
                      classifier: torch.nn.Module,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      config: dict) -> pd.DataFrame:
    """
    Orchestrator: runs compute_metrics_per_target -> saves countergan_metrics.csv,
    computes feature shift importance, class pair sensitivity, case studies, mask analysis (if available),
    and t-SNE decision boundary.
    """
    out = config.get('out_dir', '.')
    os.makedirs(out, exist_ok=True)

    print("[pipeline] Computing per-target CounterGAN metrics ...")
    metrics_df, originals_vis, cfs_vis = compute_metrics_per_target(generator, classifier, X_test, y_test, config)
    save_metrics(metrics_df, os.path.join(out, "countergan_metrics.csv"))

    # feature shift importance using the visuals gathered during metrics
    analyze_feature_shift_importance(
        originals_vis,
        cfs_vis,
        feature_names=config.get('feature_names', [f"feat_{i}" for i in range(X_test.shape[1])]),
        save_path=os.path.join(out, "feature_shift_importance.png"),
        scaler=config.get('scaler', None)
    )

    # class-pair sensitivity heatmaps
    analyze_class_pair_sensitivity(
        generator,
        classifier,
        X_test,
        y_test,
        feature_names=config.get('feature_names', [f"feat_{i}" for i in range(X_test.shape[1])]),
        config=config,
        save_dir=os.path.join(out, "class_pair_sensitivity")
    )

    # case studies (use saved visuals)
    generate_case_study_report_from_vis(
        classifier=classifier,
        originals=originals_vis,
        cfs=cfs_vis,
        config=config,
        scaler=config.get('scaler', None),
        n_samples=config.get('case_study_n', 20),
        save_dir=os.path.join(out, "case_studies")
    )

    # mask analysis (if available)
    if evaluate_pipeline_masks is not None:
        try:
            evaluate_pipeline_masks(generator, classifier, X_test, y_test, config, mode='single', compute_minimal_mask_size=True)
        except Exception as e:
            print(f"[pipeline] evaluate_pipeline_masks failed: {e}")

    # t-SNE visualization of classifier decision boundary
    try:
        plot_tsne_with_decision_boundary(
            classifier,
            X_test,
            y_test,
            save_path=os.path.join(out, "tsne_classifier_boundary.png"),
            device=config.get('cuda', 'cpu')
        )
    except Exception as e:
        print(f"[pipeline] t-SNE plotting failed: {e}")

    return metrics_df
