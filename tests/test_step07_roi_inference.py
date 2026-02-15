"""Step 7: ROI-only inference — prompt-aware local sliding windows, no full-volume."""
import json
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

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    load_pickle,
    maybe_mkdir_p,
)

from nnunetv2.inference.roi_predictor import (
    get_prompt_aware_slicers,
    nnUNetROIPredictor,
    parse_points_json,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.roi_config import load_config

STEP07_OUTPUT_DIR = "tests/outputs/step07"
FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")
DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
PREPROCESSED_DIR_LOWRES = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_lowres",
)
PREPROCESSED_DIR_FULLRES = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_fullres",
)
PLANS_PATH = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans.json",
)


def test_get_prompt_aware_slicers_small_prompt():
    """Dilated bbox < patch → 1 slicer (point at corner so dilation clamped)."""
    image_size = (32, 32, 32)
    patch_size = (16, 16, 16)
    dense_prompt = torch.zeros((1, 32, 32, 32))
    dense_prompt[0, 2, 2, 2] = 1.0
    slicers = get_prompt_aware_slicers(image_size, patch_size, 0.5, dense_prompt)
    assert len(slicers) == 1
    assert all(len(s) == 4 for s in slicers)


def test_get_prompt_aware_slicers_large_prompt():
    """bbox >= patch → multiple overlapping slicers."""
    image_size = (64, 64, 64)
    patch_size = (16, 16, 16)
    dense_prompt = torch.zeros((1, 64, 64, 64))
    dense_prompt[0, 5:45, 5:45, 5:45] = 1.0
    slicers = get_prompt_aware_slicers(image_size, patch_size, 0.5, dense_prompt)
    assert len(slicers) > 1
    assert all(len(s) == 4 for s in slicers)


def test_get_prompt_aware_slicers_empty_prompt():
    """all-zero prompt → 1 centered patch, never full sliding."""
    image_size = (32, 32, 32)
    patch_size = (16, 16, 16)
    dense_prompt = torch.zeros((1, 32, 32, 32))
    slicers = get_prompt_aware_slicers(image_size, patch_size, 0.5, dense_prompt)
    assert len(slicers) == 1
    assert slicers[0][1].stop - slicers[0][1].start <= patch_size[0]


def test_get_prompt_aware_slicers_dilated_bbox():
    """Single point at corner; dilated bbox clamped, prompt inside at least one patch."""
    image_size = (32, 32, 32)
    patch_size = (16, 16, 16)
    dense_prompt = torch.zeros((1, 32, 32, 32))
    dense_prompt[0, 2, 2, 2] = 1.0
    slicers = get_prompt_aware_slicers(image_size, patch_size, 0.5, dense_prompt)
    assert len(slicers) >= 1
    pz, py, px = 2, 2, 2
    in_any = any(
        s[1].start <= pz < s[1].stop and s[2].start <= py < s[2].stop and s[3].start <= px < s[3].stop
        for s in slicers
    )
    assert in_any, "Prompt point must be inside at least one patch"


def test_get_prompt_aware_slicers_dilation_guarantees_center():
    """Prompt at (16,16,16); at least one slicer has prompt within patch_size/2 of patch center."""
    image_size = (64, 64, 64)
    patch_size = (32, 32, 32)
    dense_prompt = torch.zeros((1, 64, 64, 64))
    dense_prompt[0, 15:18, 15:18, 15:18] = 1.0
    slicers = get_prompt_aware_slicers(image_size, patch_size, 0.5, dense_prompt)
    assert len(slicers) >= 1
    prompt_center = (16, 16, 16)
    half = 16
    found = False
    for s in slicers:
        patch_center = (
            (s[1].start + s[1].stop) // 2,
            (s[2].start + s[2].stop) // 2,
            (s[3].start + s[3].stop) // 2,
        )
        if all(abs(prompt_center[i] - patch_center[i]) <= half for i in range(3)):
            found = True
            break
    assert found, "At least one patch should have prompt near center (within patch_size/2)"


def test_parse_points_json_voxel():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"points": [[10, 20, 30], [5, 15, 25]], "points_space": "voxel"}, f)
        path = f.name
    try:
        pts, space = parse_points_json(path)
        assert space == "voxel"
        assert pts == [(10, 20, 30), (5, 15, 25)]
    finally:
        os.unlink(path)


def test_parse_points_json_world():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"points": [[100.0, 200.0, 300.0]], "points_space": "world"}, f)
        path = f.name
    try:
        pts, space = parse_points_json(path)
        assert space == "world"
        assert len(pts) == 1
        assert pts[0] == (100.0, 200.0, 300.0)
    finally:
        os.unlink(path)


def test_roi_predictor_no_full_volume_slicers():
    """ROI predictor must never call full-volume sliding."""
    pred = nnUNetROIPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False)
    with pytest.raises(RuntimeError, match="predict_logits_roi_mode"):
        pred.predict_sliding_window_return_logits(torch.randn(2, 32, 32, 32))


def _get_preprocessed_case():
    """Load one real preprocessed case. Prefer fullres, else lowres."""
    for preprocessed_dir in (PREPROCESSED_DIR_FULLRES, PREPROCESSED_DIR_LOWRES):
        if not os.path.isdir(preprocessed_dir):
            continue
        try:
            import blosc2
        except ImportError:
            continue
        ids = [
            i[:-5]
            for i in os.listdir(preprocessed_dir)
            if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")
        ]
        if not ids:
            continue
        ident = ids[0]
        data = np.array(blosc2.open(join(preprocessed_dir, ident + ".b2nd"), mode="r"))
        seg = np.array(blosc2.open(join(preprocessed_dir, ident + "_seg.b2nd"), mode="r"))
        props = load_pickle(join(preprocessed_dir, ident + ".pkl"))
        config_name = "3d_fullres" if "fullres" in preprocessed_dir else "3d_lowres"
        return data, seg, props, config_name, ident
    return None


def test_roi_predictor_end_to_end():
    """Load preprocessed case, run ROI prediction with centroid point, assert shape and valid logits."""
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    case = _get_preprocessed_case()
    if case is None:
        pytest.skip("No preprocessed data (fullres or lowres)")
    data, seg, props, config_name, _ = case
    data = torch.from_numpy(np.asarray(data, dtype=np.float32))
    cfg = load_config(FIXTURE_CONFIG)
    pm = PlansManager(PLANS_PATH)
    cm = pm.get_configuration(config_name)
    ds_json = load_json(join(os.path.dirname(PREPROCESSED_DIR_FULLRES), "dataset.json"))
    lm = pm.get_label_manager(ds_json)
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    num_in = determine_num_input_channels(pm, cm, ds_json)
    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPromptAware import (
        nnUNetTrainerPromptAware,
    )

    network = nnUNetTrainerPromptAware.build_network_architecture(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        num_in,
        lm.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = nnUNetROIPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False, device=device
    )
    pred.manual_initialization(
        network, pm, cm, [network.state_dict()], ds_json, "nnUNetTrainerPromptAware", None
    )
    shape = data.shape[1:]
    import cc3d

    seg_bin = (np.asarray(seg)[0] > 0).astype(np.uint8)
    labels = cc3d.connected_components(seg_bin)
    stats = cc3d.statistics(labels, no_slice_conversion=True)
    centroids = stats["centroids"][1:]
    points_zyx = (
        [(int(np.round(c[0])), int(np.round(c[1])), int(np.round(c[2]))) for c in centroids]
        if len(centroids) > 0
        else [(shape[0] // 2, shape[1] // 2, shape[2] // 2)]
    )
    logits = pred.predict_logits_roi_mode(data, points_zyx, props, cfg)
    assert logits.shape == (lm.num_segmentation_heads,) + shape
    assert logits.dtype == torch.float32


def _rescale_for_display(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img = np.clip((img - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img - img.min(), 0, 255).astype(np.uint8)
    return img


def test_step07_visual_output():
    """Save NIfTIs for CT viewer: full prediction, prompt heatmap."""
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    case = _get_preprocessed_case()
    if case is None:
        pytest.skip("No preprocessed data")
    data, seg, props, config_name, ident = case
    data_t = torch.from_numpy(np.asarray(data, dtype=np.float32))
    cfg = load_config(FIXTURE_CONFIG)
    pm = PlansManager(PLANS_PATH)
    cm = pm.get_configuration(config_name)
    ds_json = load_json(join(os.path.dirname(PREPROCESSED_DIR_FULLRES), "dataset.json"))
    lm = pm.get_label_manager(ds_json)
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    num_in = determine_num_input_channels(pm, cm, ds_json)
    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPromptAware import (
        nnUNetTrainerPromptAware,
    )

    network = nnUNetTrainerPromptAware.build_network_architecture(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        num_in,
        lm.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = nnUNetROIPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False, device=device
    )
    pred.manual_initialization(
        network, pm, cm, [network.state_dict()], ds_json, "nnUNetTrainerPromptAware", None
    )
    shape = data_t.shape[1:]
    import cc3d

    seg_bin = (np.asarray(seg)[0] > 0).astype(np.uint8)
    labels = cc3d.connected_components(seg_bin)
    stats = cc3d.statistics(labels, no_slice_conversion=True)
    centroids = stats["centroids"][1:]
    points_zyx = (
        [(int(np.round(c[0])), int(np.round(c[1])), int(np.round(c[2]))) for c in centroids]
        if len(centroids) > 0
        else [(shape[0] // 2, shape[1] // 2, shape[2] // 2)]
    )
    logits = pred.predict_logits_roi_mode(data_t, points_zyx, props, cfg)
    seg_pred = (logits[1] > logits[0]).numpy().astype(np.uint8)

    from nnunetv2.utilities.prompt_encoding import encode_points_to_heatmap

    prompt = encode_points_to_heatmap(
        points_zyx, shape, cfg.prompt.point_radius_vox, cfg.prompt.encoding
    ).numpy()

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = join(proj_root, STEP07_OUTPUT_DIR)
    maybe_mkdir_p(out_dir)
    plans = load_json(PLANS_PATH)
    spacing = tuple(plans["configurations"][config_name]["spacing"])
    if len(spacing) == 2:
        spacing = (1.0, spacing[0], spacing[1])

    try:
        import SimpleITK as sitk

        def _save(arr, name, dtype=None):
            arr = np.asarray(arr)
            if dtype:
                arr = arr.astype(dtype)
            itk_img = sitk.GetImageFromArray(arr)
            itk_img.SetSpacing(spacing)
            sitk.WriteImage(itk_img, join(out_dir, name))

        img_ch0 = data_t[0].numpy() if data_t.ndim == 4 else data_t.numpy()
        _save(_rescale_for_display(img_ch0), "full_image.nii.gz", np.uint8)
        _save(np.clip(prompt * 255, 0, 255).astype(np.uint8), "prompt_heatmap.nii.gz", np.uint8)
        _save(seg_pred, "full_prediction.nii.gz", np.uint8)

        from nnunetv2.inference.export_prediction import convert_preprocessed_to_original_space

        seg_orig = convert_preprocessed_to_original_space(
            seg_pred[None], props, pm, cm, is_seg=True
        )
        seg_orig_vis = np.asarray(seg_orig, dtype=np.float32)
        seg_orig_vis[seg_orig_vis < 0] = 0
        rw = pm.image_reader_writer_class()
        file_ending = ds_json.get("file_ending", ".nii.gz")
        rw.write_seg(
            seg_orig_vis.astype(np.uint8),
            join(out_dir, "full_prediction_original_space") + file_ending,
            props,
        )
    except ImportError:
        np.savez_compressed(
            join(out_dir, "step07_outputs.npz"),
            full_image=data_t.numpy(),
            prompt_heatmap=prompt,
            full_pred=seg_pred,
        )

    with open(join(out_dir, "README.txt"), "w") as f:
        f.write("Step 7 visual outputs — LesionLocator-style ROI inference\n\n")
        f.write("Preprocessed space:\n")
        f.write("  full_image.nii.gz         full preprocessed image\n")
        f.write("  prompt_heatmap.nii.gz     prompt heatmap (overlay on full_image)\n")
        f.write("  full_prediction.nii.gz    full-volume prediction\n\n")
        f.write("Original space (for CT viewer):\n")
        f.write("  full_prediction_original_space*.nii.gz — overlay on original CT\n")

    assert os.path.exists(join(out_dir, "full_prediction.nii.gz"))
    assert os.path.exists(join(out_dir, "prompt_heatmap.nii.gz"))
