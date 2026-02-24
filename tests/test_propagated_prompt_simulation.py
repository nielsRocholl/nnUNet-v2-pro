"""Tests for propagated prompt simulation (centroid + offset)."""
import os
from pathlib import Path

import numpy as np
import pytest

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.propagated_prompt_simulation import apply_propagation_offset
from nnunetv2.utilities.roi_config import load_config

FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")

DATASET013_PREPROCESSED = os.environ.get(
    "DATASET013_PREPROCESSED_DIR",
    join(
        os.environ.get("nnUNet_preprocessed", "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets/nnUNet_preprocessed"),
        "Dataset013_Longitudinal_CT",
        "nnUNetPlans_3d_fullres",
    ),
)


def test_apply_propagation_offset_bounded():
    rng = np.random.default_rng(42)
    centroid = (10, 20, 30)
    patch_shape = (64, 64, 64)
    sigma = (2.75, 5.19, 5.40)
    max_vox = 34.0
    for _ in range(100):
        out = apply_propagation_offset(centroid, patch_shape, sigma, max_vox, rng)
        assert 0 <= out[0] < patch_shape[0]
        assert 0 <= out[1] < patch_shape[1]
        assert 0 <= out[2] < patch_shape[2]
        dist = np.sqrt((out[0] - centroid[0]) ** 2 + (out[1] - centroid[1]) ** 2 + (out[2] - centroid[2]) ** 2)
        assert dist <= max_vox + 1.0


def test_apply_propagation_offset_anisotropic():
    rng = np.random.default_rng(123)
    centroid = (32, 32, 32)
    patch_shape = (64, 64, 64)
    sigma_z, sigma_y, sigma_x = 1.0, 10.0, 10.0
    sigma = (sigma_z, sigma_y, sigma_x)
    max_vox = 50.0
    offsets_z, offsets_y, offsets_x = [], [], []
    for _ in range(500):
        out = apply_propagation_offset(centroid, patch_shape, sigma, max_vox, rng)
        offsets_z.append(abs(out[0] - centroid[0]))
        offsets_y.append(abs(out[1] - centroid[1]))
        offsets_x.append(abs(out[2] - centroid[2]))
    mean_z, mean_y, mean_x = np.mean(offsets_z), np.mean(offsets_y), np.mean(offsets_x)
    assert mean_z < mean_y and mean_z < mean_x


def test_config_loads_propagated():
    cfg = load_config(FIXTURE_CONFIG)
    assert hasattr(cfg.sampling, "propagated")
    assert cfg.sampling.propagated.sigma_per_axis == (2.75, 5.19, 5.40)
    assert cfg.sampling.propagated.max_vox == 34.0


def test_config_defaults_when_propagated_absent():
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        d = {
            "prompt": {"point_radius_vox": 5, "encoding": "edt", "validation_use_prompt": True},
            "sampling": {
                "mode_probs": [0.5, 0.2, 0.15, 0.15],
                "n_spur": [1, 2],
                "n_neg": [1, 3],
                "large_lesion": {"K": 2, "K_min": 1, "K_max": 4, "max_extra": 3},
            },
            "inference": {"tile_step_size": 0.75, "disable_tta_default": False},
        }
        json.dump(d, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert hasattr(cfg.sampling, "propagated")
        assert cfg.sampling.propagated.sigma_per_axis == (2.75, 5.19, 5.40)
        assert cfg.sampling.propagated.max_vox == 34.0
    finally:
        os.unlink(path)


def test_pos_mode_prompts_differ_from_centroids():
    cfg = load_config(FIXTURE_CONFIG)
    if "nnUNet_preprocessed" not in os.environ:
        os.environ["nnUNet_preprocessed"] = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets/nnUNet_preprocessed"
    preprocessed_dir = os.environ.get("nnUNet_preprocessed", "") + "/Dataset010/nnUNetPlans_3d_fullres"
    if not os.path.isdir(preprocessed_dir):
        pytest.skip("Dataset010 preprocessed not found")
    pm = PlansManager(join(os.path.dirname(preprocessed_dir), "nnUNetPlans.json"))
    ds_json = load_json(join(os.path.dirname(preprocessed_dir), "dataset.json"))
    lm = pm.get_label_manager(ds_json)
    cm = pm.get_configuration("3d_fullres")
    patch_size = tuple(cm.patch_size)
    ds_cls = infer_dataset_class(preprocessed_dir)
    ds = ds_cls(preprocessed_dir)
    if len(ds.identifiers) == 0:
        pytest.skip("No preprocessed cases")
    dl = nnUNetPromptAwareDataLoader(
        ds, 1, patch_size, patch_size, lm, cfg,
        oversample_foreground_percent=0.8, transforms=None,
    )
    np.random.seed(456)
    found_offset = False
    for _ in range(80):
        batch = next(dl)
        if batch["data"][0, -1].max() == 0:
            continue
        seg = batch["target"][0, 0].numpy() if hasattr(batch["target"], "numpy") else batch["target"][0, 0]
        prompt = batch["data"][0, -1].numpy() if hasattr(batch["data"], "numpy") else batch["data"][0, -1]
        prompt_peak = np.unravel_index(np.argmax(prompt), prompt.shape)
        from nnunetv2.utilities.prompt_encoding import extract_centroids_from_seg
        centroids = extract_centroids_from_seg(seg)
        for c in centroids:
            dist = np.sqrt((prompt_peak[0] - c[0]) ** 2 + (prompt_peak[1] - c[1]) ** 2 + (prompt_peak[2] - c[2]) ** 2)
            if dist > 0.5:
                found_offset = True
                break
        if found_offset:
            break
    assert found_offset, "Expected at least one pos patch with prompt offset from centroid"


def test_dataset013_longitudinal_ct_real_samples():
    """Verify propagated prompt simulation on real Dataset013 Longitudinal CT samples."""
    if not os.path.isdir(DATASET013_PREPROCESSED):
        pytest.skip(
            f"Dataset013 preprocessed not found at {DATASET013_PREPROCESSED}. "
            "Set DATASET013_PREPROCESSED_DIR or nnUNet_preprocessed."
        )
    plans_path = join(os.path.dirname(DATASET013_PREPROCESSED), "nnUNetPlans.json")
    dataset_json_path = join(os.path.dirname(DATASET013_PREPROCESSED), "dataset.json")
    if not os.path.isfile(plans_path) or not os.path.isfile(dataset_json_path):
        pytest.skip("Dataset013 plans or dataset.json not found")
    cfg = load_config(FIXTURE_CONFIG)
    pm = PlansManager(plans_path)
    ds_json = load_json(dataset_json_path)
    lm = pm.get_label_manager(ds_json)
    cm = pm.get_configuration("3d_fullres")
    patch_size = tuple(cm.patch_size)
    ds_cls = infer_dataset_class(DATASET013_PREPROCESSED)
    ds = ds_cls(DATASET013_PREPROCESSED)
    if len(ds.identifiers) == 0:
        pytest.skip("No preprocessed cases in Dataset013")
    dl = nnUNetPromptAwareDataLoader(
        ds, 2, patch_size, patch_size, lm, cfg,
        oversample_foreground_percent=0.5, transforms=None,
    )
    np.random.seed(789)
    n_batches = 10
    n_with_prompt = 0
    n_with_offset = 0
    for _ in range(n_batches):
        batch = next(dl)
        for b in range(batch["data"].shape[0]):
            prompt_ch = batch["data"][b, -1]
            prompt_np = prompt_ch.numpy() if hasattr(prompt_ch, "numpy") else prompt_ch
            if prompt_np.max() <= 0:
                continue
            n_with_prompt += 1
            seg = batch["target"][b, 0]
            seg_np = seg.numpy() if hasattr(seg, "numpy") else seg
            from nnunetv2.utilities.prompt_encoding import extract_centroids_from_seg
            centroids = extract_centroids_from_seg(seg_np)
            prompt_peak = np.unravel_index(np.argmax(prompt_np), prompt_np.shape)
            for c in centroids:
                dist = np.sqrt(
                    (prompt_peak[0] - c[0]) ** 2
                    + (prompt_peak[1] - c[1]) ** 2
                    + (prompt_peak[2] - c[2]) ** 2
                )
                if dist > 0.5:
                    n_with_offset += 1
                    break
    assert n_with_prompt >= 1, "Expected at least one batch with nonzero prompt"
    assert n_with_offset >= 1, (
        "Expected propagated offset: prompt peak should differ from centroid. "
        f"Got {n_with_prompt} patches with prompt, {n_with_offset} with offset."
    )
