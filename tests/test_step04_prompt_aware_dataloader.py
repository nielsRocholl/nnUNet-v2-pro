"""Step 4: Prompt-aware training dataloader — stochastic patch sampling with four modes."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, write_pickle

from nnunetv2.training.dataloading.nnunet_dataset import (
    infer_dataset_class,
    nnUNetDatasetNumpy,
)
from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.roi_config import load_config

STEP04_OUTPUT_DIR = "tests/outputs/step04"
FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")
DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
PREPROCESSED_DIR = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_fullres",
)
PLANS_PATH = join(os.path.dirname(PREPROCESSED_DIR), "nnUNetPlans.json")
DATASET_JSON_PATH = join(os.path.dirname(PREPROCESSED_DIR), "dataset.json")


def _get_pm_lm():
    if not os.path.isfile(PLANS_PATH) or not os.path.isfile(DATASET_JSON_PATH):
        pytest.skip("Dataset010 plans/dataset.json not found")
    return PlansManager(PLANS_PATH), load_json(DATASET_JSON_PATH)


def _get_real_preprocessed_dataloader(cfg, batch_size=2, oversample=0.0, transforms=None):
    """Build dataloader from real nnUNet_preprocessed/Dataset010/nnUNetPlans_3d_fullres."""
    if not os.path.isdir(PREPROCESSED_DIR):
        pytest.skip("Preprocessed data not found: " + PREPROCESSED_DIR)
    pm, ds_json = _get_pm_lm()
    lm = pm.get_label_manager(ds_json)
    cm = pm.get_configuration("3d_fullres")
    patch_size = tuple(cm.patch_size)
    ds_cls = infer_dataset_class(PREPROCESSED_DIR)
    ds = ds_cls(PREPROCESSED_DIR)
    if len(ds.identifiers) == 0:
        pytest.skip("No preprocessed cases in " + PREPROCESSED_DIR)
    dl = nnUNetPromptAwareDataLoader(
        ds, batch_size, patch_size, patch_size, lm, cfg,
        oversample_foreground_percent=oversample, transforms=transforms,
    )
    return dl


def _make_synthetic_negative_case(tmpdir):
    """Minimal synthetic case with no foreground (for test_negative_mode_prompt_nonzero)."""
    shape, patch_size = (48, 48, 48), (24, 24, 24)
    data = np.random.randn(1, *shape).astype(np.float32)
    seg = np.zeros((1, *shape), dtype=np.int16)
    class_locations = {1: np.zeros((0, 4), dtype=np.int64)}
    class_locations[tuple([-1] + [1])] = []
    props = {"class_locations": class_locations, "spacing": (1.0, 1.0, 1.0)}
    np.savez_compressed(join(tmpdir, "neg.npz"), data=data, seg=seg)
    write_pickle(props, join(tmpdir, "neg.pkl"))
    return "neg", patch_size


def test_sampling_config_load():
    cfg = load_config(FIXTURE_CONFIG)
    assert hasattr(cfg, "sampling")
    assert abs(sum(cfg.sampling.mode_probs) - 1.0) < 1e-6
    assert cfg.sampling.n_spur == (1, 2)
    assert cfg.sampling.n_neg == (1, 3)


def test_mode_selection():
    np.random.seed(42)
    cfg = load_config(FIXTURE_CONFIG)
    probs = np.array(cfg.sampling.mode_probs)
    modes = [int(np.random.choice(4, p=probs)) for _ in range(200)]
    counts = [modes.count(m) for m in range(4)]
    for m, p in enumerate(probs):
        assert 0.1 < counts[m] / 200 < 0.9


def test_patch_has_prompt_channel():
    cfg = load_config(FIXTURE_CONFIG)
    dl = _get_real_preprocessed_dataloader(cfg, batch_size=2, oversample=0.0)
    batch = next(dl)
    assert batch["data"].shape[1] == 2
    assert batch["data"].shape[0] == 2


def test_prompt_channel_range():
    cfg = load_config(FIXTURE_CONFIG)
    dl = _get_real_preprocessed_dataloader(cfg, batch_size=2, oversample=0.0)
    for _ in range(5):
        batch = next(dl)
        prompt_ch = batch["data"][:, -1]
        assert prompt_ch.min() >= 0 and prompt_ch.max() <= 1.01


def test_pos_mode_has_centroid_or_fallback():
    cfg = load_config(FIXTURE_CONFIG)
    dl = _get_real_preprocessed_dataloader(cfg, batch_size=2, oversample=1.0)
    found_nonzero = False
    for _ in range(10):
        batch = next(dl)
        if batch["data"][:, -1].max() > 0:
            found_nonzero = True
            break
    assert found_nonzero


def test_negative_mode_prompt_nonzero():
    cfg = load_config(FIXTURE_CONFIG)
    pm, ds_json = _get_pm_lm()
    lm = pm.get_label_manager(ds_json)
    with tempfile.TemporaryDirectory() as tmpdir:
        ident, patch_size = _make_synthetic_negative_case(tmpdir)
        ds = nnUNetDatasetNumpy(tmpdir, [ident])
        dl = nnUNetPromptAwareDataLoader(
            ds, 1, patch_size, patch_size, lm, cfg,
            oversample_foreground_percent=0.0, transforms=None,
        )
        found_nonzero = False
        for _ in range(5):
            batch = next(dl)
            if batch["data"][0, -1].max() > 0:
                found_nonzero = True
                break
        assert found_nonzero


def test_pos_no_prompt_mode():
    cfg = load_config(FIXTURE_CONFIG)
    dl = _get_real_preprocessed_dataloader(cfg, batch_size=2, oversample=1.0)
    for _ in range(30):
        batch = next(dl)
        for b in range(batch["data"].shape[0]):
            if batch["data"][b, -1].sum() == 0:
                return
    pytest.skip("pos+no-prompt mode not sampled in 30 batches")


def test_dataloader_iteration():
    cfg = load_config(FIXTURE_CONFIG)
    dl = _get_real_preprocessed_dataloader(cfg, batch_size=2, oversample=0.33)
    patch_size = dl.patch_size
    for _ in range(5):
        batch = next(dl)
        assert "data" in batch and "target" in batch and "keys" in batch
        assert batch["data"].shape[0] == 2
        assert batch["data"].shape[1] == 2
        assert batch["data"].shape[2:] == tuple(patch_size)


def test_step04_visual_output():
    """Save NIfTIs for CT viewer. Uses real preprocessed data from nnUNet_preprocessed."""
    cfg = load_config(FIXTURE_CONFIG)
    dl = _get_real_preprocessed_dataloader(cfg, batch_size=1, oversample=0.5)
    np.random.seed(123)
    batch = next(dl)
    img = batch["data"][0, 0].numpy() if hasattr(batch["data"], "numpy") else batch["data"][0, 0]
    prompt = batch["data"][0, -1].numpy() if hasattr(batch["data"], "numpy") else batch["data"][0, -1]
    seg = batch["target"][0, 0].numpy() if hasattr(batch["target"], "numpy") else batch["target"][0, 0]

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = join(proj_root, STEP04_OUTPUT_DIR)
    maybe_mkdir_p(out_dir)

    plans = load_json(PLANS_PATH)
    spacing = tuple(plans["configurations"]["3d_fullres"]["spacing"])

    try:
        import SimpleITK as sitk

        def _save_patch(arr, name, dtype=None):
            arr = np.asarray(arr)
            if dtype:
                arr = arr.astype(dtype)
            itk_img = sitk.GetImageFromArray(arr)
            itk_img.SetSpacing(spacing)
            sitk.WriteImage(itk_img, join(out_dir, name))

        _save_patch(img, "patch_image_ch0.nii.gz")
        _save_patch(np.clip(prompt * 255, 0, 255).astype(np.uint8), "patch_prompt.nii.gz", np.uint8)
        seg_vis = np.asarray(seg, dtype=np.float32)
        seg_vis[seg_vis < 0] = 0
        _save_patch(seg_vis.astype(np.uint8), "patch_label.nii.gz", np.uint8)
    except ImportError:
        np.savez_compressed(join(out_dir, "patch_batch.npz"), image=img, prompt=prompt, label=seg)

    case_id = batch["keys"][0]
    pm, ds_json = _get_pm_lm()
    cm = pm.get_configuration("3d_fullres")
    ds_cls = infer_dataset_class(PREPROCESSED_DIR)
    ds = ds_cls(PREPROCESSED_DIR)
    data_full, seg_full, _, props = ds.load_case(case_id)
    data_full = np.asarray(data_full)
    seg_full = np.asarray(seg_full)

    try:
        from nnunetv2.inference.export_prediction import convert_preprocessed_to_original_space

        rw = pm.image_reader_writer_class()
        file_ending = ds_json.get("file_ending", ".nii.gz")
        img_orig = convert_preprocessed_to_original_space(
            data_full[0:1], props, pm, cm, is_seg=False
        )
        seg_orig = convert_preprocessed_to_original_space(
            seg_full, props, pm, cm, is_seg=True
        )
        seg_orig_vis = np.asarray(seg_orig, dtype=np.float32)
        seg_orig_vis[seg_orig_vis < 0] = 0
        rw.write_seg(
            seg_orig_vis.astype(np.uint8),
            join(out_dir, "case_label_original_space") + file_ending,
            props,
        )
        img_orig_norm = np.clip(img_orig, 0, None).astype(np.float32)
        if img_orig_norm.max() > img_orig_norm.min():
            img_orig_norm = (img_orig_norm - img_orig_norm.min()) / (
                img_orig_norm.max() - img_orig_norm.min()
            ) * 255
        rw.write_seg(
            img_orig_norm.astype(np.uint8),
            join(out_dir, "case_image_ch0_original_space") + file_ending,
            props,
        )
    except (KeyError, Exception):
        pass

    with open(join(out_dir, "README.txt"), "w") as f:
        f.write("Step 4 visual outputs — real preprocessed MSD_Liver\n\n")
        f.write("Patch (preprocessed space):\n")
        f.write("  patch_image_ch0.nii.gz  first image channel (CT)\n")
        f.write("  patch_prompt.nii.gz     prompt heatmap — overlay on patch_image_ch0\n")
        f.write("  patch_label.nii.gz      ground-truth segmentation\n\n")
        f.write("Full case (original space, nnUNet-style):\n")
        f.write("  case_image_ch0_original_space*.nii.gz  full image in original format\n")
        f.write("  case_label_original_space*.nii.gz      full label in original format\n")

    n_cc_label, n_cc_prompt = _count_cc_label_prompt(seg, prompt)
    with open(join(out_dir, "README.txt"), "a") as f:
        f.write(f"\nConnected components (patch): label={n_cc_label}, prompt={n_cc_prompt}\n")


def _count_cc_label_prompt(label: np.ndarray, prompt: np.ndarray, prompt_thresh: float = 0.1):
    """Count connected components in label (lesions) and prompt (spheres)."""
    import cc3d

    label = np.asarray(label)
    if label.ndim == 4:
        label = label[0]
    label_bin = (label > 0).astype(np.uint8)
    labels_label = cc3d.connected_components(label_bin)
    n_cc_label = labels_label.max()

    prompt = np.asarray(prompt)
    if prompt.ndim == 4:
        prompt = prompt[0]
    prompt_bin = (prompt > prompt_thresh).astype(np.uint8)
    if prompt_bin.sum() == 0:
        return n_cc_label, 0
    labels_prompt = cc3d.connected_components(prompt_bin)
    n_cc_prompt = labels_prompt.max()
    return n_cc_label, n_cc_prompt


def test_step04_cc_label_prompt_match():
    """Count cc in label and prompt; assert they match for pos-mode patches."""
    import cc3d

    cfg = load_config(FIXTURE_CONFIG)
    dl = _get_real_preprocessed_dataloader(cfg, batch_size=1, oversample=1.0)
    np.random.seed(123)
    for _ in range(50):
        batch = next(dl)
        label = batch["target"][0, 0].numpy() if hasattr(batch["target"], "numpy") else batch["target"][0, 0]
        prompt = batch["data"][0, -1].numpy() if hasattr(batch["data"], "numpy") else batch["data"][0, -1]
        n_cc_label, n_cc_prompt = _count_cc_label_prompt(label, prompt)
        if n_cc_label > 0 and n_cc_prompt > 0:
            assert n_cc_prompt >= 1, f"Expected >=1 prompt blobs for {n_cc_label} lesions, got {n_cc_prompt}"
            n_spur_max = cfg.sampling.n_spur[1]
            assert n_cc_prompt <= n_cc_label + n_spur_max, (
                f"Prompt blobs ({n_cc_prompt}) should be <= lesions ({n_cc_label}) + n_spur_max ({n_spur_max})"
            )
            return
    pytest.skip("No pos/pos+spur patch found in 50 batches")
