"""Test _collate_prompt_aware_outputs handles varying validation batch sizes."""
import os
from pathlib import Path

import numpy as np
import pytest

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPromptAware import (
    _collate_prompt_aware_outputs,
)

DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
PREP_999 = os.path.join(
    os.environ.get("nnUNet_preprocessed", os.path.join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset999_Merged",
)
PREP_999_FULLRES = os.path.join(PREP_999, "nnUNetPlans_3d_fullres")


def _make_val_output(batch_size: int, num_classes: int = 1) -> dict:
    """Simulate validation_step output for one batch."""
    return {
        "loss": np.float32(np.random.rand()),
        "tp_hard": np.random.randint(0, 10, (batch_size, num_classes)).astype(np.float32),
        "fp_hard": np.random.randint(0, 10, (batch_size, num_classes)).astype(np.float32),
        "fn_hard": np.random.randint(0, 10, (batch_size, num_classes)).astype(np.float32),
        "mode": np.random.randint(0, 4, batch_size, dtype=np.int32),
        "keys": [f"case_{i}" for i in range(batch_size)],
    }


def test_collate_varying_batch_sizes():
    """Collation must handle batches of size 10 and 12 (the original error case)."""
    outputs = [
        _make_val_output(10),
        _make_val_output(12),
        _make_val_output(4),
    ]
    collated = _collate_prompt_aware_outputs(outputs)
    assert collated["tp_hard"].shape == (26, 1)
    assert collated["fp_hard"].shape == (26, 1)
    assert collated["fn_hard"].shape == (26, 1)
    assert collated["mode"].shape == (26,)
    assert len(collated["keys"]) == 26
    assert len(collated["loss"]) == 3
    np.testing.assert_allclose(
        np.mean(collated["loss"]),
        np.mean([o["loss"] for o in outputs]),
        rtol=1e-5,
    )


def test_collate_varying_class_counts():
    """Collation must pad tp/fp/fn when different batches have 10 vs 12 classes (merged datasets)."""
    outputs = [
        {"loss": np.float32(0.5), "tp_hard": np.ones((4, 10)), "fp_hard": np.zeros((4, 10)),
         "fn_hard": np.zeros((4, 10)), "mode": np.zeros(4, dtype=np.int32), "keys": ["a"] * 4},
        {"loss": np.float32(0.6), "tp_hard": np.ones((2, 12)), "fp_hard": np.zeros((2, 12)),
         "fn_hard": np.zeros((2, 12)), "mode": np.ones(2, dtype=np.int32), "keys": ["b"] * 2},
    ]
    collated = _collate_prompt_aware_outputs(outputs)
    assert collated["tp_hard"].shape == (6, 12)
    assert collated["fp_hard"].shape == (6, 12)
    assert collated["fn_hard"].shape == (6, 12)
    np.testing.assert_array_equal(collated["tp_hard"][:4, 10:], 0)
    np.testing.assert_array_equal(collated["tp_hard"][4:, :], 1)


def test_collate_single_batch():
    """Single batch should collate without error."""
    outputs = [_make_val_output(2)]
    collated = _collate_prompt_aware_outputs(outputs)
    assert collated["tp_hard"].shape == (2, 1)
    assert collated["mode"].shape == (2,)


def test_collate_scalar_mode():
    """Mode can be scalar when batch_size=1."""
    outputs = [
        {"loss": np.float32(0.5), "tp_hard": np.array([[1.0]]), "fp_hard": np.array([[0.0]]),
         "fn_hard": np.array([[0.0]]), "mode": np.int32(0), "keys": ["case_0"]},
        {"loss": np.float32(0.6), "tp_hard": np.array([[2.0], [1.0]]), "fp_hard": np.array([[0.0], [0.0]]),
         "fn_hard": np.array([[0.0], [0.0]]), "mode": np.array([1, 2]), "keys": ["case_1", "case_2"]},
    ]
    collated = _collate_prompt_aware_outputs(outputs)
    assert collated["mode"].shape == (3,)
    assert collated["tp_hard"].shape == (3, 1)


@pytest.mark.skipif(
    not os.path.isdir(PREP_999_FULLRES),
    reason="Dataset999_Merged preprocessed data not found",
)
def test_collation_with_dataset999_dataloader():
    """Run dataloader on Dataset999_Merged, produce mock val outputs, collate."""
    from batchgenerators.utilities.file_and_folder_operations import join, load_json

    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.roi_config import load_config

    fixture_config = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")
    if not os.path.isfile(fixture_config):
        pytest.skip("nnunet_pro_config.json not found")
    cfg = load_config(fixture_config)

    plans_path = join(PREP_999, "nnUNetPlans.json")
    dataset_json_path = join(PREP_999, "dataset.json")
    if not os.path.isfile(plans_path) or not os.path.isfile(dataset_json_path):
        pytest.skip("Dataset999_Merged plans/dataset.json not found")

    pm = PlansManager(plans_path)
    ds_json = load_json(dataset_json_path)
    lm = pm.get_label_manager(ds_json)
    cm = pm.get_configuration("3d_fullres")
    patch_size = tuple(cm.patch_size)

    ds_cls = infer_dataset_class(PREP_999_FULLRES)
    ds = ds_cls(PREP_999_FULLRES)
    splits = load_json(join(PREP_999, "splits_final.json"))
    val_ids = splits[0]["val"]
    ds.identifiers = [i for i in ds.identifiers if i in val_ids]
    if len(ds.identifiers) == 0:
        pytest.skip("No validation cases in Dataset999_Merged")

    dl = nnUNetPromptAwareDataLoader(
        ds, batch_size=2, patch_size=patch_size, final_patch_size=patch_size,
        label_manager=lm, cfg=cfg, oversample_foreground_percent=0.0,
        force_zero_prompt=True,
    )
    it = iter(dl)
    val_outputs = []
    for _ in range(min(5, 1 + len(ds.identifiers) // 2)):
        batch = next(it)
        n = batch["data"].shape[0]
        val_outputs.append({
            "loss": np.float32(0.5),
            "tp_hard": np.random.rand(n, 1).astype(np.float32),
            "fp_hard": np.random.rand(n, 1).astype(np.float32),
            "fn_hard": np.random.rand(n, 1).astype(np.float32),
            "mode": batch["mode"],
            "keys": batch["keys"],
        })
    collated = _collate_prompt_aware_outputs(val_outputs)
    total = sum(o["tp_hard"].shape[0] for o in val_outputs)
    assert collated["tp_hard"].shape[0] == total
    assert collated["mode"].shape[0] == total
    keys_flat = collated["keys"].flatten().tolist() if isinstance(collated["keys"], np.ndarray) else collated["keys"]
    assert len(keys_flat) == total
