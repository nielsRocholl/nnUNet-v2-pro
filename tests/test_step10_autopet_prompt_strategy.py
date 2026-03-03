"""Step 10: autoPET IV prompt strategy — radius 2, two-channel encoding, get_bbox overlap."""
import os
from pathlib import Path

import numpy as np
import pytest

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.utilities.prompt_encoding import encode_points_to_heatmap_pair
from nnunetv2.utilities.roi_config import load_config

FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")


def test_encode_points_to_heatmap_pair_radius2():
    """Two-channel encoding with radius 2."""
    shape = (32, 32, 32)
    pos = [(16, 16, 16)]
    neg = [(8, 8, 8)]
    out = encode_points_to_heatmap_pair(pos, neg, shape, radius_vox=2, encoding="edt", device=None)
    assert out.shape == (2, *shape)
    assert out[0].max() > 0 and out[1].max() > 0
    assert out[0].min() >= 0 and out[0].max() <= 1.01
    assert out[1].min() >= 0 and out[1].max() <= 1.01


def test_encode_points_to_heatmap_pair_pos_only():
    """Pos only, neg channel zeros."""
    shape = (16, 16, 16)
    pos = [(8, 8, 8)]
    out = encode_points_to_heatmap_pair(pos, [], shape, radius_vox=2, encoding="binary", device=None)
    assert out.shape == (2, *shape)
    assert out[0].max() > 0
    assert out[1].max() == 0


def test_config_radius_and_mode_probs():
    """Config has radius 2, mode_probs [0.35, 0.15, 0.15, 0.35], prompt_intensity_scale 0.5."""
    cfg = load_config(FIXTURE_CONFIG)
    assert cfg.prompt.point_radius_vox == 2
    assert cfg.sampling.mode_probs == (0.35, 0.15, 0.15, 0.35)
    assert cfg.prompt.prompt_intensity_scale == 0.5


def test_encode_points_intensity_scale():
    """intensity_scale 0.5 caps heatmap max at 0.5."""
    shape = (24, 24, 24)
    pos = [(12, 12, 12)]
    out = encode_points_to_heatmap_pair(
        pos, [], shape, radius_vox=2, encoding="edt",
        device=None, intensity_scale=0.5,
    )
    assert out.shape == (2, *shape)
    assert out[0].max().item() <= 0.51
    assert out[0].max().item() > 0.4


def test_get_bbox_overlap_contains_voxel():
    """get_bbox override returns bbox that contains selected voxel (not necessarily centered)."""
    from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    import json
    cfg = load_config(FIXTURE_CONFIG)
    preprocessed_dir = os.environ.get("nnUNet_preprocessed", "") + "/Dataset010/nnUNetPlans_3d_fullres"
    if not os.path.isdir(preprocessed_dir):
        pytest.skip("Dataset010 preprocessed not found")
    plans_path = join(os.path.dirname(preprocessed_dir), "nnUNetPlans.json")
    ds_json_path = join(os.path.dirname(preprocessed_dir), "dataset.json")
    pm = PlansManager(plans_path)
    with open(ds_json_path) as f:
        ds_json = json.load(f)
    lm = pm.get_label_manager(ds_json)
    ds_cls = infer_dataset_class(preprocessed_dir)
    ds = ds_cls(preprocessed_dir)
    if len(ds.identifiers) == 0:
        pytest.skip("No preprocessed cases")
    patch_size = (32, 64, 64)
    dl = nnUNetPromptAwareDataLoader(ds, 1, patch_size, patch_size, lm, cfg, oversample_foreground_percent=0.0)
    np.random.seed(42)
    for _ in range(20):
        data, seg, _, props = ds.load_case(ds.identifiers[0])
        shape = np.array(data.shape[1:])
        cl = props.get("class_locations")
        if not cl or all(len(v) == 0 for v in cl.values() if isinstance(v, np.ndarray)):
            continue
        bbox_lbs, bbox_ubs = dl.get_bbox(shape, force_fg=True, class_locations=cl)
        voxels = np.concatenate([v for v in cl.values() if isinstance(v, np.ndarray) and len(v) > 0])
        v = voxels[np.random.randint(len(voxels))]
        for i in range(3):
            vi = int(v[i + 1])
            assert bbox_lbs[i] <= vi < bbox_ubs[i], f"Voxel {vi} not in bbox axis {i}: [{bbox_lbs[i]}, {bbox_ubs[i]})"
        return
    pytest.skip("No foreground voxels in test case")
