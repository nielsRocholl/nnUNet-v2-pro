"""Collect per-case dataset statistics during multi-dataset preprocessing."""
import numpy as np
import blosc2
from batchgenerators.utilities.file_and_folder_operations import join, isfile, save_json
from scipy.ndimage import label

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

DEFAULT_THRESHOLDS = (100, 2000, 20000)


def _trim_extremes(values: np.ndarray, trim_percentile: float) -> np.ndarray:
    if len(values) < 10 or trim_percentile <= 0:
        return values
    lo, hi = np.percentile(values, [trim_percentile * 100, (1 - trim_percentile) * 100])
    return values[(values >= lo) & (values <= hi)]


def compute_size_bin_thresholds(
    stats: dict,
    percentiles: tuple = (0.25, 0.5, 0.75),
    trim_percentile: float = 0.025,
    fallback: tuple = DEFAULT_THRESHOLDS,
) -> tuple:
    max_ccs = [s["fg_voxels"]["max_cc"] for s in stats.values() if s["fg_voxels"]["max_cc"] > 0]
    if len(max_ccs) < 10:
        return fallback
    arr = np.array(max_ccs)
    trimmed = _trim_extremes(arr, trim_percentile)
    if len(trimmed) < 5:
        return fallback
    p = np.percentile(trimmed, [x * 100 for x in percentiles])
    if not np.all(np.diff(p) > 0):
        return fallback
    return tuple(int(x) for x in p)


def get_size_bin(fg_voxels: int, thresholds: tuple = DEFAULT_THRESHOLDS) -> str:
    if fg_voxels == 0:
        return "background"
    if fg_voxels < thresholds[0]:
        return "tiny"
    if fg_voxels < thresholds[1]:
        return "small"
    if fg_voxels < thresholds[2]:
        return "medium"
    return "large"


def _load_seg(output_dir: str, case_id: str) -> np.ndarray:
    seg_path = join(output_dir, case_id + "_seg.b2nd")
    if not isfile(seg_path):
        raise FileNotFoundError(seg_path)
    arr = blosc2.open(urlpath=seg_path, mode="r")
    return np.asarray(arr)


def _fg_voxel_stats(seg: np.ndarray) -> dict:
    mask = np.asarray(seg) > 0
    total = int(mask.sum())
    if total == 0:
        return {"total": 0, "min_cc": 0, "max_cc": 0, "mean_cc": 0}
    labeled, n_cc = label(mask)
    sizes = np.bincount(labeled.ravel())[1:]
    return {
        "total": total,
        "min_cc": int(sizes.min()),
        "max_cc": int(sizes.max()),
        "mean_cc": float(sizes.mean()),
    }


def _resolve_dataset(case_id: str, dataset_json: dict, dataset_name: str) -> str:
    source_datasets = dataset_json.get("source_datasets")
    if not source_datasets:
        return dataset_name
    best = ""
    for s in source_datasets:
        if case_id.startswith(s + "_") and len(s) > len(best):
            best = s
    return best if best else dataset_name


def collect_case_statistics(
    output_dir: str,
    dataset_json: dict,
    dataset_name: str,
    size_bin_thresholds: tuple = DEFAULT_THRESHOLDS,
    size_bins_config: dict | None = None,
) -> dict:
    identifiers = nnUNetDatasetBlosc2.get_identifiers(output_dir)
    stats = {}
    for case_id in identifiers:
        seg = _load_seg(output_dir, case_id)
        fg = _fg_voxel_stats(seg)
        dataset = _resolve_dataset(case_id, dataset_json, dataset_name)
        stats[case_id] = {"dataset": dataset, "fg_voxels": fg}

    if size_bins_config and size_bins_config.get("mode") == "percentile":
        thresholds = compute_size_bin_thresholds(
            stats,
            percentiles=tuple(size_bins_config.get("percentiles", [0.25, 0.5, 0.75])),
            trim_percentile=float(size_bins_config.get("trim_percentile", 0.025)),
        )
    elif size_bins_config and size_bins_config.get("mode") == "fixed":
        thresholds = tuple(size_bins_config.get("thresholds", DEFAULT_THRESHOLDS))
    else:
        thresholds = size_bin_thresholds

    for case_id in stats:
        stats[case_id]["size_bin"] = get_size_bin(stats[case_id]["fg_voxels"]["max_cc"], thresholds)
    stats["_thresholds_used"] = thresholds
    return stats


def save_case_stats(
    stats: dict,
    output_path: str,
    size_bin_thresholds: tuple = DEFAULT_THRESHOLDS,
    size_bins_config: dict | None = None,
) -> None:
    thresholds = stats.pop("_thresholds_used", None) or size_bin_thresholds
    meta = {"size_bin_thresholds": list(thresholds)}
    if size_bins_config:
        meta["size_bin_mode"] = size_bins_config.get("mode", "fixed")
        meta["size_bin_config"] = {k: v for k, v in size_bins_config.items() if k != "mode"}
    out = {"_metadata": meta, **stats}
    save_json(out, output_path, sort_keys=False)
