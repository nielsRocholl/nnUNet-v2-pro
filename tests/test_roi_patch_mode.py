"""Per-point patch ROI helpers and merge safety."""
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.inference.roi_predictor import (
    background_logits_vector,
    centered_spatial_slices_at_point,
    map_points_zyx_unpadded_to_padded,
    safe_divide_merged_logits,
    shift_spatial_slices,
    spatial_slices_to_tuple,
    touching_patch_faces_from_logits,
)
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.roi_config import InferenceConfig, load_config

FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")


def test_inference_config_defaults_from_partial_json():
    cfg = load_config(FIXTURE_CONFIG)
    assert cfg.inference.roi_inference_mode == "dilated_sliding"
    assert cfg.inference.max_patch_expansion_visits == 64
    assert cfg.inference.max_patch_expansion_depth is None


def test_inference_config_per_point_keys():
    import json
    import tempfile

    with open(FIXTURE_CONFIG) as f:
        d = json.load(f)
    d["inference"] = {
        **d.get("inference", {}),
        "roi_inference_mode": "per_point_patch",
        "max_patch_expansion_visits": 8,
        "max_patch_expansion_depth": 2,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(d, tf)
        path = tf.name
    try:
        cfg = load_config(path)
        assert cfg.inference.roi_inference_mode == "per_point_patch"
        assert cfg.inference.max_patch_expansion_visits == 8
        assert cfg.inference.max_patch_expansion_depth == 2
    finally:
        import os

        os.unlink(path)


def test_map_points_unpadded_to_padded():
    rev = (slice(None), slice(2, 34), slice(1, 65), slice(0, 64))
    pts = [(5, 10, 20)]
    out = map_points_zyx_unpadded_to_padded(pts, rev)
    assert out == [(7, 11, 20)]


def test_centered_spatial_clamped():
    ps = (16, 16, 16)
    sh = (20, 20, 20)
    sz, sy, sx = centered_spatial_slices_at_point(0, 0, 0, ps, sh)
    assert sz.start == 0 and sz.stop == 16
    sz2, sy2, sx2 = centered_spatial_slices_at_point(19, 19, 19, ps, sh)
    assert sz2.stop == 20 and sz2.start == 4


def test_shift_spatial_clamped():
    ps = (8, 8, 8)
    sh = (16, 16, 16)
    sz, sy, sx = slice(4, 12), slice(0, 8), slice(0, 8)
    nz, ny, nx = shift_spatial_slices(sz, sy, sx, 0, True, ps, sh)
    assert nz.start <= sz.start


def test_touching_faces_low_z():
    lm = LabelManager(label_dict={"background": 0, 1: 1}, regions_class_order=None)
    logits = torch.zeros(2, 8, 8, 8)
    logits[1, 0, 2:6, 2:6] = 8.0
    faces = touching_patch_faces_from_logits(logits, lm)
    assert (0, True) in faces


def test_safe_divide_no_nan():
    lm = LabelManager(label_dict={"background": 0, 1: 1}, regions_class_order=None)
    dev = torch.device("cpu")
    bg = background_logits_vector(lm, 2, dev)
    pred = torch.zeros(2, 4, 4, 4)
    pred[1, 1:3, 1:3, 1:3] = 3.0
    n = torch.zeros(4, 4, 4)
    n[1:3, 1:3, 1:3] = 2.0
    safe_divide_merged_logits(pred, n, bg)
    assert not torch.isnan(pred).any()
    assert pred[1, 0, 0, 0] == bg[1]


def test_spatial_slices_tuple_stable():
    sz, sy, sx = slice(0, 8), slice(2, 10), slice(4, 12)
    assert spatial_slices_to_tuple(sz, sy, sx) == (0, 8, 2, 10, 4, 12)


def test_inference_config_dataclass_replaceable():
    a = InferenceConfig(0.5, False)
    assert a.roi_inference_mode == "dilated_sliding"
    from dataclasses import replace

    b = replace(a, roi_inference_mode="per_point_patch", max_patch_expansion_visits=3)
    assert b.roi_inference_mode == "per_point_patch"
    assert b.max_patch_expansion_visits == 3
