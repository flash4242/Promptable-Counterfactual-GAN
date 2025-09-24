"""
Evaluation utilities focused on diagnosing mask usefulness for CounterGAN.

Key capabilities:
- Systematic evaluation of *explicit* mask sets (no hidden random masks unless requested).
- Single-feature evaluation (all unit masks).
- Exhaustive combinations up to a small k (with safeguards against combinatorial explosion).
- Evaluation of a custom list of masks (useful for semantic/grouped masks).
- Optional per-sample minimal-mask-size computation (which mask size first achieves a flip).
- Aggregation into mask-level and feature-level DataFrames and CSV output.

Design principles:
- Deterministic by default (seeded when sampling required).
- Memory-conscious: evaluates masks in chunks across batches rather than materializing everything at once.
- Reproducible and research-friendly outputs (CSV + pandas DataFrames).

Usage (high level):
- import evaluate_mask_suite and call evaluate_pipeline_masks(...)
- Use config dict to control device, batch_size, chunk sizes and combinatorics limits.

"""

import os
import math
import itertools
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def _mask_array_to_str(mask: np.ndarray) -> str:
    return ''.join(['1' if m else '0' for m in mask.astype(int)])


def generate_single_feature_masks(num_features: int) -> List[np.ndarray]:
    """Return list of unit masks (each mask allows exactly one feature)."""
    return [np.eye(1, num_features, k=i, dtype=int).reshape(-1) for i in range(num_features)]


def generate_all_combinations_masks(num_features: int, max_k: int,
                                    max_allowed: int = 2000) -> List[np.ndarray]:
    """Generate all combinations of features up to size max_k.

    Raises a ValueError if the number of generated masks would exceed max_allowed
    to avoid combinatorial blowup. You can lower max_allowed in config to be
    stricter.
    """
    masks = []
    total = 0
    for k in range(1, max_k + 1):
        c = math.comb(num_features, k)
        total += c
        if total > max_allowed:
            raise ValueError(f"Too many masks would be generated ({total}). Increase 'max_allowed' or lower 'max_k'.")
        for comb in itertools.combinations(range(num_features), k):
            m = np.zeros(num_features, dtype=int)
            m[list(comb)] = 1
            masks.append(m)
    return masks


def sample_random_masks(num_features: int, num_masks: int, sizes: Optional[Sequence[int]] = None,
                        seed: Optional[int] = None) -> List[np.ndarray]:
    """Sample masks randomly. If sizes is provided, it's used as candidate mask sizes
    (will be sampled uniformly among the provided sizes)."""
    rng = np.random.RandomState(seed)
    masks = []
    if sizes is None:
        # allow a range from 1..num_features but bias towards small masks
        sizes = list(range(1, min(5, num_features) + 1))
    for _ in range(num_masks):
        k = int(rng.choice(sizes))
        idx = rng.choice(num_features, size=k, replace=False)
        m = np.zeros(num_features, dtype=int)
        m[idx] = 1
        masks.append(m)
    return masks


def _masks_list_to_tensor(masks: List[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(masks, axis=0).astype(np.float32)  # (M, F)
    return torch.from_numpy(arr).to(device)


def evaluate_masks_for_targets(
    generator: torch.nn.Module,
    classifier: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    masks: List[np.ndarray],
    config: Dict,
    targets: Optional[Sequence[int]] = None,
    compute_minimal_mask_size: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Evaluate a given list of masks for each target class.

    Args:
        generator, classifier: torch modules (already trained).
        X, y: numpy arrays (N x F) and (N,).
        masks: list of binary numpy arrays of length F. Each mask indicates which
               features are allowed to change (1=modifiable).
        config: dictionary with keys:
            - 'cuda' : torch.device or string
            - 'batch_size'
            - 'chunk_size_masks' : how many masks to evaluate in parallel per batch iteration (memory/speed tradeoff)
            - 'out_dir' : directory to save results
        targets: explicit list of target classes to evaluate; default = all classes
        compute_minimal_mask_size: if True, compute per-sample minimal mask size that achieves flip w.r.t each target.

    Returns:
        mask_results_df: DataFrame with one row per (target, mask) and aggregated metrics.
        feature_results_df: DataFrame with per-feature aggregated statistics (marginal contributions etc.)
        minimal_mask_df (optional): per-target per-sample minimal mask size summary (if requested)
    """
    device = config['cuda']
    batch_size = int(config.get('batch_size', 256))
    chunk_size_masks = int(config.get('chunk_size_masks', 8))
    compute_minimal = compute_minimal_mask_size

    assert isinstance(masks, list) and len(masks) > 0, "Provide an explicit list of masks to evaluate."
    num_features = X.shape[1]
    M = len(masks)

    # sanity check mask shapes
    for i, m in enumerate(masks):
        if m.shape[0] != num_features:
            raise ValueError(f"Mask {i} has length {m.shape[0]} but expected {num_features}")

    # Dataset with indices so we can map per-sample results (needed when computing minimal mask sizes)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    idx_t = torch.arange(len(X), dtype=torch.long)
    dataset = TensorDataset(X_t, y_t, idx_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # convert masks to tensor on device for faster ops when chunking
    masks_tensor = _masks_list_to_tensor(masks, device)  # (M, F)

    generator.eval()
    classifier.eval()

    # results containers per target
    mask_rows = []  # will collect dicts for DataFrame

    # prepare per-feature aggregation storage across targets+mask evaluations
    # We'll aggregate later from the mask-level DataFrame.

    # Optional minimal mask size per sample per target
    minimal_mask_records = []  # list of dicts (target, sample_idx, minimal_size)

    # Loop over targets
    unique_classes = np.unique(y)
    if targets is None:
        targets = list(unique_classes.astype(int))

    with torch.no_grad():
        for target in targets:
            # We'll keep per-mask accumulators (sums) so we can compute means at the end
            total_attempts = np.zeros(M, dtype=np.int64)
            total_successes = np.zeros(M, dtype=np.int64)
            total_pred_gain = np.zeros(M, dtype=np.float64)  # sum of (p_cf_target - p_orig_target)
            total_l1 = np.zeros(M, dtype=np.float64)  # sum of L1 residual norms
            total_feat_abs = np.zeros((M, num_features), dtype=np.float64)  # sum of abs changes per feature

            # if requested, minimal size per sample (initialize +inf -> will become integer sizes)
            if compute_minimal:
                minimal_size = np.full(len(X), np.inf, dtype=np.float32)

            # Iterate over dataset batches
            for x_batch, y_batch, idx_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                idx_batch = idx_batch.numpy()

                # Only evaluate samples whose original class != target
                keep_mask = (y_batch != target)
                if keep_mask.sum() == 0:
                    continue

                x_valid = x_batch[keep_mask]
                idx_valid = idx_batch[keep_mask.cpu().numpy()]

                bs_valid = x_valid.size(0)
                # precompute original classifier probabilities for this valid batch
                logits_orig = classifier(x_valid)
                probs_orig = F.softmax(logits_orig, dim=1).cpu().numpy()  # (bs_valid, C)
                p_orig_target = probs_orig[:, int(target)]  # (bs_valid,)

                # target onehot same for all valid samples in this batch
                target_vec = torch.full((bs_valid,), int(target), device=device, dtype=torch.long)
                target_onehot = F.one_hot(target_vec, int(unique_classes.size)).float()

                # evaluate masks in chunks to balance memory/time
                for start in range(0, M, chunk_size_masks):
                    end = min(M, start + chunk_size_masks)
                    c = end - start
                    masks_chunk = masks_tensor[start:end]  # (c, F)

                    # repeat x_valid and target_onehot for each mask in chunk
                    x_rep = x_valid.repeat(c, 1)  # (c*bs_valid, F)
                    target_rep = target_onehot.repeat(c, 1)  # (c*bs_valid, C)
                    # mask_rep should repeat each mask 'bs_valid' times in block order
                    mask_rep = masks_chunk.repeat_interleave(bs_valid, dim=0)  # (c*bs_valid, F)

                    residuals = generator(x_rep, target_rep, mask=mask_rep)
                    cfs = x_rep + residuals

                    logits_cf = classifier(cfs)
                    probs_cf = F.softmax(logits_cf, dim=1).cpu().numpy()  # (c*bs_valid, C)
                    p_cf_target = probs_cf[:, int(target)]

                    # reshape back to (c, bs_valid, ...)
                    p_cf_target = p_cf_target.reshape(c, bs_valid)
                    # compute predicted classes
                    preds_cf = logits_cf.argmax(dim=1).cpu().numpy().reshape(c, bs_valid)

                    # l1 residual norms and feature abs changes
                    residuals_cpu = residuals.cpu().numpy().reshape(c, bs_valid, num_features)
                    l1_per_example = np.abs(residuals_cpu).sum(axis=2)  # (c, bs_valid)
                    feat_abs_per_example = np.abs(residuals_cpu)  # (c, bs_valid, F)

                    # mask sizes for chunk
                    mask_sizes = masks_chunk.sum(dim=1).cpu().numpy().astype(int)  # (c,)

                    # accumulate
                    for j in range(c):
                        midx = start + j
                        attempts = bs_valid
                        successes = (preds_cf[j] == int(target)).sum()

                        total_attempts[midx] += attempts
                        total_successes[midx] += int(successes)

                        # pred gain sum
                        total_pred_gain[midx] += float((p_cf_target[j] - p_orig_target).sum())

                        # l1 sums
                        total_l1[midx] += float(l1_per_example[j].sum())

                        # feature abs sums
                        total_feat_abs[midx] += feat_abs_per_example[j].sum(axis=0)

                        # update minimal size per sample if requested
                        if compute_minimal:
                            # for those examples that succeeded in this mask, update minimal_size[idx] if mask_sizes[j] smaller
                            succeeded_mask = (preds_cf[j] == int(target))  # boolean array length bs_valid
                            if succeeded_mask.any():
                                idxs_succeeded = idx_valid[succeeded_mask]
                                cur_min = minimal_size[idxs_succeeded]
                                minimal_size[idxs_succeeded] = np.minimum(cur_min, mask_sizes[j])

            # After iterating batches, compute aggregated metrics for this target
            for m_idx, m in enumerate(masks):
                attempts = int(total_attempts[m_idx])
                successes = int(total_successes[m_idx])
                success_rate = float(successes / attempts) if attempts > 0 else float('nan')
                avg_pred_gain = float(total_pred_gain[m_idx] / attempts) if attempts > 0 else float('nan')
                avg_l1 = float(total_l1[m_idx] / attempts) if attempts > 0 else float('nan')
                avg_feat_abs = (total_feat_abs[m_idx] / attempts) if attempts > 0 else np.full(num_features, np.nan)

                mask_row = {
                    'target': int(target),
                    'mask_idx': int(m_idx),
                    'mask_bits': _mask_array_to_str(np.array(m, dtype=int)),
                    'mask_size': int(np.array(m, dtype=int).sum()),
                    'attempts': attempts,
                    'successes': successes,
                    'success_rate': success_rate,
                    'avg_pred_gain': avg_pred_gain,
                    'avg_l1_residual': avg_l1,
                }

                # add per-feature mean abs change as columns prefixed with f0_, f1_, ...
                for fi in range(num_features):
                    mask_row[f'f{fi}_mean_abs_change'] = float(avg_feat_abs[fi])

                mask_rows.append(mask_row)

            # record minimal size distribution for this target (if computed)
            if compute_minimal:
                # minimal_size is np.array of length N; inf means never succeeded for that sample
                # create a small summary as DataFrame rows per sample where original class != target
                valid_idx_global = np.where(y != target)[0]
                minimal_vals = minimal_size[valid_idx_global]
                for sample_idx, val in zip(valid_idx_global, minimal_vals):
                    rec = {'target': int(target), 'sample_idx': int(sample_idx)}
                    if np.isfinite(val):
                        rec['minimal_mask_size'] = int(val)
                    else:
                        rec['minimal_mask_size'] = int(0)  # use 0 to mark "no successful mask"
                    minimal_mask_records.append(rec)

    mask_results_df = pd.DataFrame(mask_rows)

    # Feature-level aggregation (marginal contributions)
    # For each feature i compute:
    #  - included_attempts/successes: aggregates over masks where feature is present
    #  - excluded_attempts/successes: aggregates over masks where feature is absent
    feature_agg = []
    if not mask_results_df.empty:
        for fi in range(num_features):
            included = mask_results_df[mask_results_df['mask_bits'].str.get(fi) == '1'] if isinstance(mask_results_df['mask_bits'].iloc[0], str) else None

            # safer approach: parse mask bits into boolean array
            mask_bits_matrix = np.stack([list(map(int, list(s))) for s in mask_results_df['mask_bits'].values], axis=0)
            included_mask = mask_bits_matrix[:, fi] == 1

            incl_df = mask_results_df.iloc[included_mask]
            excl_df = mask_results_df.iloc[~included_mask]

            incl_attempts = incl_df['attempts'].sum()
            incl_successes = incl_df['successes'].sum()
            excl_attempts = excl_df['attempts'].sum()
            excl_successes = excl_df['successes'].sum()

            incl_rate = float(incl_successes / incl_attempts) if incl_attempts > 0 else float('nan')
            excl_rate = float(excl_successes / excl_attempts) if excl_attempts > 0 else float('nan')

            # average mean abs change for this feature when included
            f_col = f'f{fi}_mean_abs_change'
            mean_abs_when_included = float(incl_df[f_col].mean()) if not incl_df.empty else float('nan')

            feature_agg.append({
                'feature_idx': fi,
                'included_attempts': int(incl_attempts),
                'included_successes': int(incl_successes),
                'included_success_rate': incl_rate,
                'excluded_attempts': int(excl_attempts),
                'excluded_successes': int(excl_successes),
                'excluded_success_rate': excl_rate,
                'marginal_success_increase': (incl_rate - excl_rate) if (not np.isnan(incl_rate) and not np.isnan(excl_rate)) else float('nan'),
                'mean_abs_change_when_included': mean_abs_when_included,
            })

    feature_results_df = pd.DataFrame(feature_agg)

    minimal_df = pd.DataFrame(minimal_mask_records) if compute_minimal else None

    return mask_results_df, feature_results_df, minimal_df


def save_mask_evaluation(mask_df: pd.DataFrame, feature_df: pd.DataFrame, minimal_df: Optional[pd.DataFrame], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    mask_path = os.path.join(out_dir, 'mask_level_results.csv')
    feat_path = os.path.join(out_dir, 'feature_level_results.csv')
    mask_df.to_csv(mask_path, index=False)
    feature_df.to_csv(feat_path, index=False)
    if minimal_df is not None:
        minimal_path = os.path.join(out_dir, 'minimal_mask_sizes_per_sample.csv')
        minimal_df.to_csv(minimal_path, index=False)
    print(f"Saved mask-level results to {mask_path}")
    print(f"Saved feature-level results to {feat_path}")
    if minimal_df is not None:
        print(f"Saved minimal-mask-size per-sample summary to {minimal_path}")


def evaluate_pipeline_masks(
    generator: torch.nn.Module,
    classifier: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: Dict,
    mode: str = 'single',
    max_k: int = 2,
    custom_masks: Optional[List[Sequence[int]]] = None,
    sample_random_count: int = 0,
    compute_minimal_mask_size: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    High-level orchestrator that builds a mask set according to 'mode' and runs evaluate_masks_for_targets.

    Modes:
      - 'single' : evaluate all single-feature masks (default)
      - 'combinatorial' : evaluate all combos up to size 'max_k' (careful: combinatorial)
      - 'custom' : evaluate a user-supplied list of masks (custom_masks should be provided)
      - 'sampled' : sample 'sample_random_count' random masks (for exploratory analysis)

    Returns dict with keys: 'mask_df', 'feature_df', 'minimal_df' (if requested)
    """
    num_features = X.shape[1]

    if mode == 'single':
        masks = generate_single_feature_masks(num_features)
    elif mode == 'combinatorial':
        masks = generate_all_combinations_masks(num_features, max_k, max_allowed=config.get('max_allowed_masks', 2000))
    elif mode == 'custom':
        if custom_masks is None:
            raise ValueError("custom_masks must be provided when mode='custom'")
        # normalize custom masks to numpy arrays of length num_features
        masks = []
        for m in custom_masks:
            arr = np.array(m, dtype=int)
            if arr.shape[0] != num_features:
                raise ValueError("Each custom mask must have length equal to number of features")
            masks.append(arr)
    elif mode == 'sampled':
        if sample_random_count <= 0:
            raise ValueError("sample_random_count must be > 0 when mode='sampled'")
        sizes = config.get('sample_sizes', None)
        masks = sample_random_masks(num_features, sample_random_count, sizes=sizes, seed=config.get('seed', 42))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mask_df, feature_df, minimal_df = evaluate_masks_for_targets(
        generator, classifier, X, y, masks, config, targets=None, compute_minimal_mask_size=compute_minimal_mask_size
    )

    save_mask_evaluation(mask_df, feature_df, minimal_df, config.get('out_dir_mask', './mask_eval_out'))

    out = {'mask_df': mask_df, 'feature_df': feature_df}
    if minimal_df is not None:
        out['minimal_df'] = minimal_df
    return out


# End of file
