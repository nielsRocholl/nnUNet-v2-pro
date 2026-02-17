"""Step 6: Trainer variant — wire prompt-aware dataloader, keep Dice+CE loss."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_json, write_pickle

from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class, nnUNetDatasetNumpy
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPromptAware import nnUNetTrainerPromptAware
from nnunetv2.utilities.large_lesion_sampling import is_large_lesion
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.prompt_encoding import build_prompt_channel, extract_centroids_from_seg
from nnunetv2.utilities.roi_config import load_config

STEP06_OUTPUT_DIR = "tests/outputs/step06"
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


def _get_preprocessed_dir_and_config():
    for cfg_name, preprocessed_dir in [("3d_fullres", PREPROCESSED_DIR_FULLRES), ("3d_lowres", PREPROCESSED_DIR_LOWRES)]:
        if os.path.isdir(preprocessed_dir):
            return preprocessed_dir, cfg_name
    return None, None


def _ensure_splits_for_single_sample():
    """Create splits_final.json for 1-sample dataset (required for do_split)."""
    preprocessed_dir, _ = _get_preprocessed_dir_and_config()
    if preprocessed_dir is None:
        return
    ds_cls = infer_dataset_class(preprocessed_dir)
    ds = ds_cls(preprocessed_dir)
    if len(ds.identifiers) < 2:
        base = os.path.dirname(preprocessed_dir)
        splits_file = join(base, "splits_final.json")
        if not os.path.isfile(splits_file):
            ids = list(ds.identifiers)
            splits = [{"train": ids, "val": ids}] * 5
            save_json(splits, splits_file)


def _get_trainer(device=None, config_path=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_path is None:
        config_path = FIXTURE_CONFIG
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    preprocessed_dir, config_name = _get_preprocessed_dir_and_config()
    if preprocessed_dir is None:
        pytest.skip("No preprocessed data (fullres or lowres) found")
    ds = infer_dataset_class(preprocessed_dir)(preprocessed_dir)
    if len(ds.identifiers) == 0:
        pytest.skip("No preprocessed cases")
    plans = load_json(PLANS_PATH)
    dataset_json = load_json(join(os.path.dirname(preprocessed_dir), "dataset.json"))
    return nnUNetTrainerPromptAware(
        plans=plans,
        configuration=config_name,
        fold=0,
        dataset_json=dataset_json,
        device=device,
        config_path=config_path,
    )


def test_trainer_uses_default_config_when_none():
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    plans = load_json(PLANS_PATH)
    dataset_json = load_json(join(os.path.dirname(PREPROCESSED_DIR_FULLRES), "dataset.json"))
    trainer = nnUNetTrainerPromptAware(
        plans=plans,
        configuration="3d_fullres",
        fold=0,
        dataset_json=dataset_json,
        config_path=None,
    )
    assert trainer.roi_cfg is not None
    assert trainer.roi_cfg.prompt.encoding == "edt"


def test_trainer_instantiation():
    trainer = _get_trainer()
    assert hasattr(trainer, "roi_cfg")
    assert trainer.roi_cfg.prompt.point_radius_vox == 5
    pm = trainer.plans_manager
    cm = pm.get_configuration(trainer.configuration_name)
    num_in = trainer.plans_manager.get_label_manager(trainer.dataset_json).num_segmentation_heads
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    num_in = determine_num_input_channels(pm, cm, trainer.dataset_json)
    net = nnUNetTrainerPromptAware.build_network_architecture(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        num_in,
        trainer.label_manager.num_segmentation_heads,
        True,
    )
    first_conv = next(
        m for m in net.modules()
        if hasattr(m, "weight") and m.weight.dim() == 5 and m.weight.shape[1] == num_in + 1
    )
    assert first_conv.weight.shape[1] == num_in + 1


def test_get_dataloaders():
    _ensure_splits_for_single_sample()
    trainer = _get_trainer()
    trainer.initialize()
    dl_tr, dl_val = trainer.get_dataloaders()
    batch_tr = next(dl_tr)
    batch_val = next(dl_val)
    assert "data" in batch_tr and "target" in batch_tr
    assert "data" in batch_val and "target" in batch_val
    num_img_channels = trainer.num_input_channels
    assert batch_tr["data"].shape[1] == num_img_channels + 1
    assert batch_val["data"].shape[1] == num_img_channels + 1


def test_one_training_step():
    _ensure_splits_for_single_sample()
    trainer = _get_trainer()
    trainer.initialize()
    dl_tr, _ = trainer.get_dataloaders()
    batch = next(dl_tr)
    data = batch["data"].to(trainer.device)
    target = batch["target"]
    if isinstance(target, list):
        target = [t.to(trainer.device) for t in target]
    else:
        target = target.to(trainer.device)
    trainer.optimizer.zero_grad()
    out = trainer.network(data)
    if isinstance(out, tuple):
        loss = trainer.loss(out, target)
    else:
        loss = trainer.loss(out, target)
    loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
    assert np.isfinite(loss_val)
    loss.backward()
    for p in trainer.network.parameters():
        if p.requires_grad and p.grad is not None:
            assert p.grad.abs().sum() > 0
            break


def test_validation_batch_zero_prompt():
    _ensure_splits_for_single_sample()
    trainer = _get_trainer()
    trainer.initialize()
    _, dl_val = trainer.get_dataloaders()
    use_prompt = trainer.roi_cfg.prompt.validation_use_prompt
    for _ in range(5):
        batch = next(dl_val)
        prompt_ch = batch["data"][:, -1]
        arr = prompt_ch.numpy() if hasattr(prompt_ch, "numpy") else np.asarray(prompt_ch)
        if not use_prompt:
            assert arr.max() == 0 and arr.min() == 0
        else:
            assert arr.shape[-3:] == tuple(trainer.configuration_manager.patch_size)


def _make_real_case_with_artificial_large_lesion(tmpdir, data, seg, props, patch_size):
    """Real image + artificial large lesion mask in same space."""
    seg = np.asarray(seg)
    if seg.ndim == 4:
        seg = seg[0]
    shape = seg.shape
    z0, z1 = max(0, shape[0] // 4), min(shape[0], shape[0] // 4 + patch_size[0] + 10)
    y0, y1 = max(0, shape[1] // 4), min(shape[1], shape[1] // 4 + patch_size[1] + 10)
    x0, x1 = max(0, shape[2] // 4), min(shape[2], shape[2] // 4 + patch_size[2] + 10)
    seg_mod = np.zeros((1,) + shape, dtype=np.int16)
    seg_mod[0, z0:z1, y0:y1, x0:x1] = 1
    bbox = (z0, z1, y0, y1, x0, x1)
    assert is_large_lesion(bbox, patch_size), "Artificial lesion must be larger than patch"
    zs, ys, xs = np.where(seg_mod[0] > 0)
    n_fg = min(1000, len(zs))
    idx = np.random.choice(len(zs), n_fg, replace=False)
    class_locations = {
        1: np.column_stack([np.zeros(n_fg), zs[idx], ys[idx], xs[idx]]).astype(np.int64)
    }
    class_locations[tuple([-1] + [1])] = []
    props_mod = dict(props)
    props_mod["class_locations"] = class_locations
    np.savez_compressed(join(tmpdir, "large.npz"), data=data, seg=seg_mod)
    write_pickle(props_mod, join(tmpdir, "large.pkl"))
    return "large"


def test_large_lesion_path():
    preprocessed_dir, config_name = _get_preprocessed_dir_and_config()
    if preprocessed_dir is None:
        pytest.skip("No preprocessed data")
    cfg = load_config(FIXTURE_CONFIG)
    pm = PlansManager(PLANS_PATH)
    cm = pm.get_configuration(config_name)
    patch_size = tuple(cm.patch_size)
    ds_cls = infer_dataset_class(preprocessed_dir)
    ds = ds_cls(preprocessed_dir)
    if len(ds.identifiers) == 0:
        pytest.skip("No preprocessed cases")
    case_id = ds.identifiers[0]
    data, seg, seg_prev, props = ds.load_case(case_id)
    data = np.asarray(data)
    seg = np.asarray(seg)
    with tempfile.TemporaryDirectory() as tmpdir:
        ident = _make_real_case_with_artificial_large_lesion(tmpdir, data, seg, props, patch_size)
        ds_json = load_json(join(os.path.dirname(preprocessed_dir), "dataset.json"))
        lm = pm.get_label_manager(ds_json)
        ds_mod = nnUNetDatasetNumpy(tmpdir, [ident])
        from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader

        dl = nnUNetPromptAwareDataLoader(
            ds_mod, 1, patch_size, patch_size, lm, cfg,
            oversample_foreground_percent=0.0, transforms=None,
        )
        found_extras = False
        for _ in range(30):
            batch = next(dl)
            if batch["data"].shape[0] > 1:
                found_extras = True
                break
        assert found_extras


def _rescale_for_display(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img = np.clip((img - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img - img.min(), 0, 255).astype(np.uint8)
    return img


def _save_forced_correct_prompt(out_dir, trainer, spacing, config_name, _save, _rescale_for_display):
    """Save patch with forced correct prompt: 1 sphere per connected component in label."""
    preprocessed_dir, _ = _get_preprocessed_dir_and_config()
    if preprocessed_dir is None:
        return
    cfg = load_config(FIXTURE_CONFIG)
    pm = PlansManager(PLANS_PATH)
    cm = pm.get_configuration(config_name)
    patch_size = tuple(cm.patch_size)
    ds_cls = infer_dataset_class(preprocessed_dir)
    ds = ds_cls(preprocessed_dir)
    if len(ds.identifiers) == 0:
        return
    case_id = ds.identifiers[0]
    data, seg, seg_prev, props = ds.load_case(case_id)
    data = np.asarray(data)
    seg = np.asarray(seg)
    if seg.ndim == 4:
        seg = seg[0]
    shape = seg.shape
    centroids = extract_centroids_from_seg(seg[None] if seg.ndim == 3 else seg)
    if len(centroids) == 0:
        return
    cz = int(np.median([c[0] for c in centroids]))
    cy = int(np.median([c[1] for c in centroids]))
    cx = int(np.median([c[2] for c in centroids]))
    bbox_lbs = [
        max(0, cz - patch_size[0] // 2),
        max(0, cy - patch_size[1] // 2),
        max(0, cx - patch_size[2] // 2),
    ]
    bbox_ubs = [
        min(shape[0], bbox_lbs[0] + patch_size[0]),
        min(shape[1], bbox_lbs[1] + patch_size[1]),
        min(shape[2], bbox_lbs[2] + patch_size[2]),
    ]
    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
    data_crop = np.asarray(crop_and_pad_nd(data, bbox, 0))
    seg_crop = np.asarray(crop_and_pad_nd(seg[None] if seg.ndim == 3 else seg, bbox, -1))
    patch_slices = (
        slice(bbox_lbs[0], bbox_ubs[0]),
        slice(bbox_lbs[1], bbox_ubs[1]),
        slice(bbox_lbs[2], bbox_ubs[2]),
    )
    seg_full = seg[None] if seg.ndim == 3 else seg
    prompt = build_prompt_channel(seg_full, patch_slices, cfg)
    prompt_np = prompt.squeeze(0).numpy()
    seg_vis = np.asarray(seg_crop[0] if seg_crop.ndim == 4 else seg_crop, dtype=np.float32)
    seg_vis[seg_vis < 0] = 0
    if data_crop.ndim == 4:
        img_crop = data_crop[0]
    else:
        img_crop = data_crop
    _save(_rescale_for_display(img_crop), "correct_prompt_image.nii.gz", np.uint8)
    _save(np.clip(prompt_np * 255, 0, 255).astype(np.uint8), "correct_prompt_prompt.nii.gz", np.uint8)
    _save(seg_vis.astype(np.uint8), "correct_prompt_label.nii.gz", np.uint8)

    import cc3d
    n_cc_label = cc3d.connected_components((seg_vis > 0).astype(np.uint8)).max()
    n_cc_prompt = cc3d.connected_components((prompt_np > 0.1).astype(np.uint8)).max()
    assert n_cc_prompt <= n_cc_label, f"Prompt spheres ({n_cc_prompt}) must be <= label CCs ({n_cc_label})"
    if n_cc_label >= 1:
        assert n_cc_prompt >= 1, f"Expected >=1 sphere when label has {n_cc_label} CCs"


def test_step06_visual_output():
    _ensure_splits_for_single_sample()
    trainer = _get_trainer()
    trainer.initialize()
    dl_tr, dl_val = trainer.get_dataloaders()
    np.random.seed(123)
    batch_tr = next(dl_tr)
    batch_val = next(dl_val)

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = join(proj_root, STEP06_OUTPUT_DIR)
    maybe_mkdir_p(out_dir)

    plans = load_json(PLANS_PATH)
    config_name = trainer.configuration_name
    spacing = tuple(plans["configurations"][config_name]["spacing"])

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    idx = 0
    target_tr = batch_tr["target"]
    target_val = batch_val["target"]
    seg_tr = _to_np(target_tr[0][idx, 0]) if isinstance(target_tr, list) else _to_np(target_tr[idx, 0])
    seg_val = _to_np(target_val[0][idx, 0]) if isinstance(target_val, list) else _to_np(target_val[idx, 0])
    img_tr = _to_np(batch_tr["data"][idx, 0])
    prompt_tr = _to_np(batch_tr["data"][idx, -1])
    img_val = _to_np(batch_val["data"][idx, 0])
    prompt_val = _to_np(batch_val["data"][idx, -1])

    try:
        import SimpleITK as sitk

        def _save(arr, name, dtype=None):
            arr = np.asarray(arr)
            if dtype:
                arr = arr.astype(dtype)
            itk_img = sitk.GetImageFromArray(arr)
            itk_img.SetSpacing(spacing)
            sitk.WriteImage(itk_img, join(out_dir, name))

        _save(_rescale_for_display(img_tr), "patch_image.nii.gz", np.uint8)
        _save(np.clip(prompt_tr * 255, 0, 255).astype(np.uint8), "patch_prompt.nii.gz", np.uint8)
        seg_tr_vis = np.asarray(seg_tr, dtype=np.float32)
        seg_tr_vis[seg_tr_vis < 0] = 0
        _save(seg_tr_vis.astype(np.uint8), "patch_label.nii.gz", np.uint8)
        _save(_rescale_for_display(img_val), "val_patch_image.nii.gz", np.uint8)
        _save(np.clip(prompt_val * 255, 0, 255).astype(np.uint8), "val_patch_prompt.nii.gz", np.uint8)
        seg_val_vis = np.asarray(seg_val, dtype=np.float32)
        seg_val_vis[seg_val_vis < 0] = 0
        _save(seg_val_vis.astype(np.uint8), "val_patch_label.nii.gz", np.uint8)
    except ImportError:
        np.savez_compressed(
            join(out_dir, "step06_batch.npz"),
            img_tr=img_tr, prompt_tr=prompt_tr, seg_tr=seg_tr,
            img_val=img_val, prompt_val=prompt_val, seg_val=seg_val,
        )

    _save_forced_correct_prompt(out_dir, trainer, spacing, config_name, _save, _rescale_for_display)

    with open(join(out_dir, "README.txt"), "w") as f:
        f.write("Step 6 visual outputs — nnUNetTrainerPromptAware\n\n")
        f.write("Train batch (random mode):\n")
        f.write("  patch_image.nii.gz   image channel (rescaled)\n")
        f.write("  patch_prompt.nii.gz prompt heatmap — overlay on patch_image\n")
        f.write("  patch_label.nii.gz  ground-truth segmentation\n\n")
        f.write("Forced correct prompt (1 sphere per CC):\n")
        f.write("  correct_prompt_image.nii.gz  image crop\n")
        f.write("  correct_prompt_prompt.nii.gz centroids encoded — one sphere per lesion CC\n")
        f.write("  correct_prompt_label.nii.gz  segmentation crop\n\n")
        f.write("Validation batch:\n")
        f.write("  val_patch_image.nii.gz   image channel\n")
        f.write("  val_patch_prompt.nii.gz  prompt (zero if validation_use_prompt=false)\n")
        f.write("  val_patch_label.nii.gz   ground-truth segmentation\n")
