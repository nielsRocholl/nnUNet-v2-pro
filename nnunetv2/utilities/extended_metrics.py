"""Extended Dice metrics for wandb: per-patch, per-dataset, per-size-bin."""
from collections import defaultdict
from typing import List, Optional

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile


def _dice_from_tp_fp_fn(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    denom = 2 * tp + fp + fn
    dc = np.where(denom > 0, 2 * tp / denom, np.nan)
    return float(np.nanmean(dc))


def compute_extended_dice_metrics(
    tp_flat: np.ndarray,
    fp_flat: np.ndarray,
    fn_flat: np.ndarray,
    keys_flat: List[str],
    case_stats_path: Optional[str],
    dataset_name: str,
) -> dict:
    """
    Returns {dice_per_patch, dice_per_dataset: {...}, dice_per_size_bin: {...}}.
    Per-dataset and per-size-bin are empty if case_stats missing.
    """
    denom = 2 * tp_flat + fp_flat + fn_flat
    dice_per_patch_arr = np.where(denom > 0, 2 * tp_flat / denom, np.nan)
    mean_per_patch = np.nanmean(dice_per_patch_arr, axis=1)
    dice_per_patch = float(np.nanmean(mean_per_patch))

    result = {"dice_per_patch": dice_per_patch, "dice_per_dataset": {}, "dice_per_size_bin": {}}

    if not isfile(case_stats_path or ""):
        return result

    case_stats = load_json(case_stats_path)
    dataset_to_idx = defaultdict(list)
    bin_to_idx = defaultdict(list)

    for i, cid in enumerate(keys_flat):
        info = case_stats.get(cid)
        if info is None or cid == "_metadata":
            ds = dataset_name
            bin_name = "unknown"
        else:
            ds = info.get("dataset", dataset_name)
            bin_name = info.get("size_bin", "unknown")
        dataset_to_idx[ds].append(i)
        bin_to_idx[bin_name].append(i)

    for ds, idx in dataset_to_idx.items():
        tp_d = np.sum(tp_flat[idx], axis=0)
        fp_d = np.sum(fp_flat[idx], axis=0)
        fn_d = np.sum(fn_flat[idx], axis=0)
        result["dice_per_dataset"][ds] = _dice_from_tp_fp_fn(tp_d, fp_d, fn_d)

    for bin_name, idx in bin_to_idx.items():
        tp_b = np.sum(tp_flat[idx], axis=0)
        fp_b = np.sum(fp_flat[idx], axis=0)
        fn_b = np.sum(fn_flat[idx], axis=0)
        result["dice_per_size_bin"][bin_name] = _dice_from_tp_fp_fn(tp_b, fp_b, fn_b)

    return result
