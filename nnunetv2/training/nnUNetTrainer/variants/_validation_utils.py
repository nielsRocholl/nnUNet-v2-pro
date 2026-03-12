"""Helpers for prompt-aware validation: DDP gather and extended metrics."""
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from torch import distributed as dist

from nnunetv2.training.dataloading.prompt_aware_data_loader import (
    MODE_NEG,
    MODE_POS,
    MODE_POS_NO_PROMPT,
    MODE_POS_SPUR,
)
from nnunetv2.utilities.extended_metrics import compute_extended_dice_metrics

MODE_NAMES = {MODE_POS: "pos", MODE_POS_SPUR: "pos_spur", MODE_POS_NO_PROMPT: "pos_no_prompt", MODE_NEG: "neg"}


def gather_validation_outputs_ddp(
    tp_flat: np.ndarray,
    fp_flat: np.ndarray,
    fn_flat: np.ndarray,
    keys_flat: List[str],
    mode_flat: np.ndarray,
    loss: Union[list, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, float]:
    """Gather validation outputs across DDP ranks. Returns flattened arrays and mean loss."""
    world_size = dist.get_world_size()
    tps = [None] * world_size
    fps = [None] * world_size
    fns = [None] * world_size
    keys_list = [None] * world_size
    modes_list = [None] * world_size
    dist.all_gather_object(tps, tp_flat)
    dist.all_gather_object(fps, fp_flat)
    dist.all_gather_object(fns, fn_flat)
    dist.all_gather_object(keys_list, keys_flat)
    dist.all_gather_object(modes_list, mode_flat.tolist())
    tp_flat = np.vstack(tps)
    fp_flat = np.vstack(fps)
    fn_flat = np.vstack(fns)
    keys_flat = [k for kl in keys_list for k in kl]
    mode_flat = np.array([m for ml in modes_list for m in ml])
    losses_val = [None] * world_size
    dist.all_gather_object(losses_val, loss)
    loss_here = float(np.concatenate([np.atleast_1d(l) for l in losses_val]).mean())
    return tp_flat, fp_flat, fn_flat, keys_flat, mode_flat, loss_here


def log_dice_by_mode(
    tp_flat: np.ndarray,
    fp_flat: np.ndarray,
    fn_flat: np.ndarray,
    mode_flat: np.ndarray,
    logger: Any,
    current_epoch: int,
) -> None:
    """Log val_Dice_{pos,pos_spur,pos_no_prompt,neg} and mean_fg_dice to logger."""
    for m, name in MODE_NAMES.items():
        idx = mode_flat == m
        if not np.any(idx):
            logger.log(f"val_Dice_{name}", float("nan"), current_epoch)
            continue
        tp_m = np.sum(tp_flat[idx], axis=0)
        fp_m = np.sum(fp_flat[idx], axis=0)
        fn_m = np.sum(fn_flat[idx], axis=0)
        dc = np.array(
            [2 * t / (2 * t + p + n) if (2 * t + p + n) > 0 else np.nan for t, p, n in zip(tp_m, fp_m, fn_m)]
        )
        logger.log(f"val_Dice_{name}", float(np.nanmean(dc)), current_epoch)
    tp_global = np.sum(tp_flat, axis=0)
    fp_global = np.sum(fp_flat, axis=0)
    fn_global = np.sum(fn_flat, axis=0)
    dc_global = np.array(
        [2 * t / (2 * t + p + n) if (2 * t + p + n) > 0 else np.nan for t, p, n in zip(tp_global, fp_global, fn_global)]
    )
    logger.log("mean_fg_dice", float(np.nanmean(dc_global)), current_epoch)
    logger.log("dice_per_class_or_region", dc_global.tolist(), current_epoch)


def build_wandb_extended_metrics(
    tp_flat: np.ndarray,
    fp_flat: np.ndarray,
    fn_flat: np.ndarray,
    keys_flat: List[str],
    case_stats_path: str,
    dataset_name: str,
) -> Dict[str, float]:
    """Build wandb dict with extended Dice metrics (per-patch, per-dataset, per-size-bin)."""
    keys_for_ext = keys_flat if len(keys_flat) == len(tp_flat) else []
    ext = compute_extended_dice_metrics(
        tp_flat, fp_flat, fn_flat, keys_for_ext, case_stats_path, dataset_name
    )
    wb_dict = {"dice_per_patch": ext["dice_per_patch"]}
    for ds, d in ext["dice_per_dataset"].items():
        wb_dict[f"dice_dataset_{ds}"] = d
    for bin_name, d in ext["dice_per_size_bin"].items():
        wb_dict[f"dice_size_bin_{bin_name}"] = d
    return wb_dict
