"""Single-patch inference optimizations: fg from logits, one-tile fast path vs legacy merge."""
import numpy as np
import torch

from nnunetv2.inference.roi_predictor import (
    _patch_fg_bool_from_logits,
    background_logits_vector,
    centered_spatial_slices_at_point,
    safe_divide_merged_logits,
)
from nnunetv2.utilities.inference_execution import env_wants_torch_compile, inference_accumulator_dtype
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def test_patch_fg_matches_convert_logits_labels():
    lm = LabelManager({"background": 0, "a": 1, "b": 2}, regions_class_order=None, force_use_labels=True)
    logits = torch.randn(3, 10, 11, 12)
    ref = lm.convert_logits_to_segmentation(logits)
    ref_np = ref.numpy() if isinstance(ref, torch.Tensor) else ref
    fg_ref = np.zeros(ref_np.shape, dtype=bool)
    for fl in lm.foreground_labels:
        fg_ref |= ref_np == fl
    fg_new = _patch_fg_bool_from_logits(logits, lm)
    assert fg_new.shape == fg_ref.shape
    assert np.array_equal(fg_ref, fg_new)


def test_patch_fg_matches_convert_logits_regions():
    lm = LabelManager(
        {"background": 0, "organ": (1, 2)},
        regions_class_order=[1],
        force_use_labels=False,
    )
    logits = torch.randn(1, 8, 8, 8)
    ref = lm.convert_logits_to_segmentation(logits)
    ref_np = ref.numpy() if isinstance(ref, torch.Tensor) else np.asarray(ref)
    fg_ref = ref_np > 0
    fg_new = _patch_fg_bool_from_logits(logits, lm)
    assert np.array_equal(fg_ref, fg_new)


def test_single_tile_merged_equals_pred_raw():
    """One overlapping tile: Gaussian-weighted accumulate then divide yields raw network logits in the patch."""
    lm = LabelManager({"background": 0, "lesion": 1}, regions_class_order=None, force_use_labels=True)
    num_heads = 2
    patch_size = (8, 8, 8)
    padded_shape = (20, 20, 20)
    pz, py, px = 10, 10, 10
    sz, sy, sx = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
    pred_raw = torch.randn(num_heads, *patch_size)
    gaussian = torch.abs(torch.randn(patch_size)) + 0.5
    dev = torch.device("cpu")
    predicted = torch.zeros((num_heads, *padded_shape), dtype=torch.float32)
    n_pred = torch.zeros(padded_shape, dtype=torch.float32)
    predicted[(slice(None), sz, sy, sx)] += pred_raw * gaussian
    n_pred[sz, sy, sx] += gaussian
    bg = background_logits_vector(lm, num_heads, dev, dtype=torch.float32)
    safe_divide_merged_logits(predicted, n_pred, bg)
    merged = predicted[(slice(None), sz, sy, sx)]
    assert torch.allclose(merged, pred_raw, rtol=1e-4, atol=1e-4)

    bg_plane = bg.view(-1, 1, 1, 1).expand(num_heads, *padded_shape).clone()
    fast = pred_raw.clone()
    out = bg_plane.clone()
    out[(slice(None), sz, sy, sx)] = fast
    assert torch.allclose(out[(slice(None), sz, sy, sx)], merged, rtol=1e-5, atol=1e-5)


def test_inference_accumulator_dtype_cpu_forces_float32():
    assert inference_accumulator_dtype(torch.device("cpu")) == torch.float32


def test_env_wants_torch_compile_smoke():
    assert isinstance(env_wants_torch_compile(), bool)
