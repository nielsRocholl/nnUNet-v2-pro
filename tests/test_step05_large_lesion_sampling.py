"""Step 5: Large-lesion add-on — sparse extra patches for lesions larger than patch."""
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

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class, nnUNetDatasetNumpy
from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
from nnunetv2.utilities.large_lesion_sampling import (
    get_lesion_bboxes_zyx,
    is_large_lesion,
    sample_extra_centers_for_large_lesion,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.roi_config import load_config

STEP05_OUTPUT_DIR = "tests/outputs/step05"
FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")
DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
PREPROCESSED_DIR_FULLRES = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_fullres",
)
PREPROCESSED_DIR_LOWRES = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_lowres",
)
PLANS_PATH = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans.json",
)


def test_config_load():
    cfg = load_config(FIXTURE_CONFIG)
    assert hasattr(cfg.sampling, "large_lesion")
    ll = cfg.sampling.large_lesion
    assert ll.K == (2, 2)
    assert ll.K_min == 1
    assert ll.K_max == 4
    assert ll.max_extra == 3


def test_get_lesion_bboxes_zyx():
    seg = np.zeros((1, 40, 50, 60), dtype=np.uint8)
    seg[0, 5:15, 10:25, 20:40] = 1
    seg[0, 25:35, 30:45, 5:25] = 1
    bboxes = get_lesion_bboxes_zyx(seg)
    assert len(bboxes) == 2
    for bbox in bboxes:
        assert len(bbox) == 6
        zmin, zmax, ymin, ymax, xmin, xmax = bbox
        assert zmax > zmin and ymax > ymin and xmax > xmin


def test_is_large_lesion():
    patch = (48, 192, 192)
    assert is_large_lesion((0, 60, 0, 200, 0, 200), patch)
    assert is_large_lesion((0, 40, 0, 250, 0, 100), patch)
    assert not is_large_lesion((0, 40, 0, 100, 0, 100), patch)


def test_is_large_lesion_small():
    patch = (48, 192, 192)
    assert not is_large_lesion((0, 40, 0, 100, 0, 100), patch)


def test_sample_extra_centers():
    cfg = load_config(FIXTURE_CONFIG).sampling.large_lesion
    shape = (80, 256, 256)
    seg = np.zeros((1,) + shape, dtype=np.uint8)
    seg[0, 10:70, 20:240, 20:240] = 1
    bbox = (10, 70, 20, 240, 20, 240)
    patch_size = (48, 192, 192)
    np.random.seed(42)
    centers = sample_extra_centers_for_large_lesion(seg, bbox, patch_size, cfg)
    assert len(centers) <= cfg.K_max
    for cz, cy, cx in centers:
        assert 0 <= cz < shape[0] and 0 <= cy < shape[1] and 0 <= cx < shape[2]
        assert seg[0, cz, cy, cx] > 0


def _make_synthetic_large_lesion_case(tmpdir, shape, patch_size, lesion_slice):
    data = np.random.randn(1, *shape).astype(np.float32)
    seg = np.zeros((1, *shape), dtype=np.int16)
    seg[0][lesion_slice] = 1
    zs, ys, xs = np.where(seg[0] > 0)
    n_fg = min(1000, len(zs))
    idx = np.random.choice(len(zs), n_fg, replace=False)
    class_locations = {1: np.column_stack([np.zeros(n_fg), zs[idx], ys[idx], xs[idx]]).astype(np.int64)}
    class_locations[tuple([-1] + [1])] = []
    props = {"class_locations": class_locations, "spacing": (1.0, 1.0, 1.0)}
    np.savez_compressed(join(tmpdir, "large.npz"), data=data, seg=seg)
    write_pickle(props, join(tmpdir, "large.pkl"))
    return "large", patch_size


def test_dataloader_adds_extra_patches():
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    cfg = load_config(FIXTURE_CONFIG)
    patch_size = (48, 192, 192)
    shape = (80, 256, 256)
    lesion_slice = (slice(10, 70), slice(20, 240), slice(20, 240))
    with tempfile.TemporaryDirectory() as tmpdir:
        ident, _ = _make_synthetic_large_lesion_case(tmpdir, shape, patch_size, lesion_slice)
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        pm = PlansManager(PLANS_PATH)
        ds_json = load_json(join(os.path.dirname(PREPROCESSED_DIR_FULLRES), "dataset.json"))
        lm = pm.get_label_manager(ds_json)
        ds = nnUNetDatasetNumpy(tmpdir, [ident])
        dl = nnUNetPromptAwareDataLoader(
            ds, 1, patch_size, patch_size, lm, cfg,
            oversample_foreground_percent=0.0, transforms=None,
        )
        found_extras = False
        for _ in range(20):
            batch = next(dl)
            n = batch["data"].shape[0]
            if n > 1:
                found_extras = True
                assert batch["data"].shape[1] == 2
                assert batch["target"].shape[0] == n
                for b in range(n):
                    assert batch["data"][b, -1].min() >= 0 and batch["data"][b, -1].max() <= 1.01
                break
        assert found_extras


def _get_real_preprocessed_dataloader_step05(cfg, config_name="3d_lowres", batch_size=1):
    """Build dataloader from real nnUNet_preprocessed/Dataset010 (lowres or fullres)."""
    preprocessed_dir = (
        PREPROCESSED_DIR_LOWRES if config_name == "3d_lowres" else PREPROCESSED_DIR_FULLRES
    )
    if not os.path.isdir(preprocessed_dir):
        pytest.skip(f"Preprocessed data not found: {preprocessed_dir}")
    pm = PlansManager(PLANS_PATH)
    ds_json = load_json(join(os.path.dirname(preprocessed_dir), "dataset.json"))
    lm = pm.get_label_manager(ds_json)
    cm = pm.get_configuration(config_name)
    patch_size = tuple(cm.patch_size)
    ds_cls = infer_dataset_class(preprocessed_dir)
    ds = ds_cls(preprocessed_dir)
    if len(ds.identifiers) == 0:
        pytest.skip(f"No preprocessed cases in {preprocessed_dir}")
    dl = nnUNetPromptAwareDataLoader(
        ds, batch_size, patch_size, patch_size, lm, cfg,
        oversample_foreground_percent=0.5, transforms=None,
    )
    return dl, pm, ds_json, ds, config_name


def _rescale_for_display(img: np.ndarray) -> np.ndarray:
    """Rescale preprocessed (normalized) CT to 0-255 for viewer display."""
    img = np.asarray(img, dtype=np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img = np.clip((img - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img - img.min(), 0, 255).astype(np.uint8)
    return img


def test_step05_visual_output():
    """Save NIfTIs for CT viewer. Uses real preprocessed data from nnUNet_preprocessed/Dataset010."""
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    cfg = load_config(FIXTURE_CONFIG)
    plans = load_json(PLANS_PATH)

    batch, ds, config_name = None, None, None
    for cfg_name in ("3d_lowres", "3d_fullres"):
        try:
            dl, _, _, ds, config_name = _get_real_preprocessed_dataloader_step05(
                cfg, config_name=cfg_name
            )
        except Exception:
            continue
        np.random.seed(123)
        for _ in range(50):
            batch = next(dl)
            if batch["data"].shape[0] > 1:
                break
        if batch is not None:
            break
    if batch is None:
        pytest.skip("No preprocessed data available")
    spacing = tuple(plans["configurations"][config_name]["spacing"])

    idx = 1 if batch["data"].shape[0] > 1 else 0
    img = batch["data"][idx, 0].numpy() if hasattr(batch["data"], "numpy") else batch["data"][idx, 0]
    prompt = batch["data"][idx, -1].numpy() if hasattr(batch["data"], "numpy") else batch["data"][idx, -1]
    seg = batch["target"][idx, 0].numpy() if hasattr(batch["target"], "numpy") else batch["target"][idx, 0]
    case_id = batch["keys"][idx]

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = join(proj_root, STEP05_OUTPUT_DIR)
    maybe_mkdir_p(out_dir)

    try:
        import SimpleITK as sitk

        def _save(arr, name, dtype=None):
            arr = np.asarray(arr)
            if dtype:
                arr = arr.astype(dtype)
            itk_img = sitk.GetImageFromArray(arr)
            itk_img.SetSpacing(spacing)
            sitk.WriteImage(itk_img, join(out_dir, name))

        img_display = _rescale_for_display(img)
        _save(img_display, "extra_patch_0_image.nii.gz", np.uint8)
        _save(np.clip(prompt * 255, 0, 255).astype(np.uint8), "extra_patch_0_prompt.nii.gz", np.uint8)
        seg_vis = np.asarray(seg, dtype=np.float32)
        seg_vis[seg_vis < 0] = 0
        _save(seg_vis.astype(np.uint8), "extra_patch_0_label.nii.gz", np.uint8)

        data_full, seg_full, _, props = ds.load_case(case_id)
        data_full = np.asarray(data_full)
        seg_full = np.asarray(seg_full)
        large_mask = (seg_full > 0).any(axis=0).astype(np.uint8) if seg_full.ndim == 4 else (seg_full > 0).astype(np.uint8)
        _save(large_mask, "large_lesion_mask.nii.gz", np.uint8)

        try:
            from nnunetv2.inference.export_prediction import convert_preprocessed_to_original_space

            pm = PlansManager(PLANS_PATH)
            ds_json = load_json(join(os.path.dirname(PREPROCESSED_DIR_FULLRES), "dataset.json"))
            cm = pm.get_configuration(config_name)
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
    except ImportError:
        np.savez_compressed(
            join(out_dir, "step05_batch.npz"),
            image=img, prompt=prompt, label=seg,
        )
    patch_type = "extra (large-lesion)" if idx == 1 else "main"
    with open(join(out_dir, "README.txt"), "w") as f:
        f.write("Step 5 visual outputs — real preprocessed data from nnUNet_preprocessed/Dataset010\n\n")
        f.write("Patch (preprocessed space):\n")
        f.write("  extra_patch_0_image.nii.gz  — image channel (rescaled), patch_type=%s\n" % patch_type)
        f.write("  extra_patch_0_prompt.nii.gz — prompt heatmap (overlay on image)\n")
        f.write("  extra_patch_0_label.nii.gz  — ground-truth segmentation\n")
        f.write("  large_lesion_mask.nii.gz   — full-volume lesion mask\n\n")
        f.write("Full case (original space, nnUNet-style — use for CT viewer):\n")
        f.write("  case_image_ch0_original_space*.nii.gz  full image, proper orientation/spacing\n")
        f.write("  case_label_original_space*.nii.gz      full label, overlay on case_image\n")
