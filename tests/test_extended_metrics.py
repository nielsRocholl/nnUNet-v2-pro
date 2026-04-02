"""Tests for extended wandb metrics: per-patch, per-dataset, per-size-bin Dice."""
import os

import numpy as np
import pytest
import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json

from nnunetv2.utilities.extended_metrics import compute_extended_dice_metrics

try:
    from tests.conftest import NNUNET_DUMMY_BASE as DUMMY_BASE
except ImportError:
    DUMMY_BASE = None


def test_dice_per_patch_no_case_stats():
    n_patches, n_classes = 10, 2
    tp = np.random.rand(n_patches, n_classes).astype(np.float64) * 10
    fp = np.random.rand(n_patches, n_classes).astype(np.float64) * 5
    fn = np.random.rand(n_patches, n_classes).astype(np.float64) * 5
    keys = [f"case_{i}" for i in range(n_patches)]

    out = compute_extended_dice_metrics(tp, fp, fn, keys, None, "Dataset999")
    assert "dice_per_patch" in out
    assert 0 <= out["dice_per_patch"] <= 1 or np.isnan(out["dice_per_patch"])
    assert out["dice_per_dataset"] == {}
    assert out["dice_per_size_bin"] == {}


def test_dice_per_patch_perfect():
    tp = np.array([[5.0, 5.0], [10.0, 10.0]])
    fp = fn = np.zeros_like(tp)
    keys = ["c1", "c2"]

    out = compute_extended_dice_metrics(tp, fp, fn, keys, None, "DS")
    assert out["dice_per_patch"] == 1.0


def test_dice_per_patch_vs_aggregated():
    tp = np.array([[2.0, 2.0], [2.0, 2.0]])
    fp = fn = np.array([[1.0, 1.0], [1.0, 1.0]])
    keys = ["c1", "c2"]

    out = compute_extended_dice_metrics(tp, fp, fn, keys, None, "DS")
    dice_per_patch = 2 * 2 / (2 * 2 + 1 + 1)
    assert abs(out["dice_per_patch"] - dice_per_patch) < 1e-6


def test_dice_per_dataset_and_size_bin(tmp_path):
    case_stats = {
        "c1": {"dataset": "D1", "size_bin": "small", "fg_voxels": {}},
        "c2": {"dataset": "D1", "size_bin": "large", "fg_voxels": {}},
        "c3": {"dataset": "D2", "size_bin": "small", "fg_voxels": {}},
    }
    stats_path = tmp_path / "case_stats.json"
    save_json({**case_stats, "_metadata": {}}, str(stats_path), sort_keys=False)

    tp = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    fp = fn = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    keys = ["c1", "c2", "c3"]

    out = compute_extended_dice_metrics(tp, fp, fn, keys, str(stats_path), "DS")
    assert "D1" in out["dice_per_dataset"]
    assert "D2" in out["dice_per_dataset"]
    assert "small" in out["dice_per_size_bin"]
    assert "large" in out["dice_per_size_bin"]


def test_missing_case_stats_path():
    out = compute_extended_dice_metrics(
        np.ones((2, 1)), np.zeros((2, 1)), np.zeros((2, 1)),
        ["a", "b"], "/nonexistent/path.json", "DS",
    )
    assert out["dice_per_dataset"] == {}
    assert out["dice_per_size_bin"] == {}


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_preprocessed_dataset():
    prep = os.environ.get("nnUNet_preprocessed")
    if prep is None and DUMMY_BASE is not None:
        prep = join(DUMMY_BASE, "nnUNet_preprocessed")
    if prep is None:
        return None, None
    for name in ("Dataset999_Merged", "Dataset004_Hippocampus", "Dataset010", "Dataset999_IntegrationTest_Hippocampus"):
        base = join(prep, name)
        plans = join(base, "nnUNetPlans.json")
        fullres = join(base, "nnUNetPlans_3d_fullres")
        if os.path.isfile(plans) and os.path.isdir(fullres):
            return name, "3d_fullres"
    return None, None


@pytest.mark.slow
def test_extended_metrics_training_one_epoch():
    """Run 1 epoch with nnUNetTrainerPromptAware to verify extended metrics path."""
    dataset_name, config = _find_preprocessed_dataset()
    if dataset_name is None:
        pytest.skip("No preprocessed dataset (004, 010, or 999) found")

    from nnunetv2.run.run_training import run_training

    run_training(
        dataset_name_or_id=dataset_name,
        configuration=config,
        fold=0,
        device=_device(),
        trainer_class_name="nnUNetTrainerPromptAware",
        num_epochs=1,
        use_wandb=False,
    )
