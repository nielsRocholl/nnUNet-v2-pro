"""Border expansion planning: perimeter-connected shell components -> extra patch centers."""
import numpy as np

from nnunetv2.inference.roi_predictor import (
    centered_spatial_slices_at_point,
    plan_border_expansion_centers_from_fg,
    spatial_slices_to_tuple,
)


def test_plan_two_disjoint_low_x_face_blobs():
    dz = dy = dx = 64
    fg = np.zeros((dz, dy, dx), dtype=bool)
    fg[30:36, 10:16, 0] = True
    fg[30:36, 40:46, 0] = True
    patch_size = (dz, dy, dx)
    padded_shape = (128, 128, 128)
    sz = sy = sx = slice(32, 32 + 64)
    seed_key = spatial_slices_to_tuple(sz, sy, sx)
    plans = plan_border_expansion_centers_from_fg(
        fg, sz, sy, sx, patch_size, padded_shape, seed_key, max_centers=16, quant_vox=1
    )
    keys = set()
    for pz, py, px in plans:
        u, v, w = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
        keys.add(spatial_slices_to_tuple(u, v, w))
    assert len(keys) == len(plans)
    assert len(plans) == 2


def test_plan_single_blob_one_center():
    dz = dy = dx = 32
    fg = np.zeros((dz, dy, dx), dtype=bool)
    fg[0, 8:14, 8:14] = True
    patch_size = (dz, dy, dx)
    padded_shape = (96, 96, 96)
    sz = sy = sx = slice(16, 16 + 32)
    seed_key = spatial_slices_to_tuple(sz, sy, sx)
    plans = plan_border_expansion_centers_from_fg(
        fg, sz, sy, sx, patch_size, padded_shape, seed_key, max_centers=8, quant_vox=1
    )
    assert len(plans) == 1


def test_plan_one_shell_cc_both_low_and_high_z_faces():
    dz = dy = dx = 16
    fg = np.zeros((dz, dy, dx), dtype=bool)
    fg[0, :, :] = True
    fg[dz - 1, :, :] = True
    fg[:, 0, 0] = True
    patch_size = (dz, dy, dx)
    padded_shape = (64, 64, 64)
    sz = sy = sx = slice(8, 8 + 16)
    seed_key = spatial_slices_to_tuple(sz, sy, sx)
    plans = plan_border_expansion_centers_from_fg(
        fg, sz, sy, sx, patch_size, padded_shape, seed_key, max_centers=8, quant_vox=1
    )
    keys = set()
    for pz, py, px in plans:
        u, v, w = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
        keys.add(spatial_slices_to_tuple(u, v, w))
    assert len(keys) == len(plans)
    assert len(plans) >= 2
    pz_set = {p[0] for p in plans}
    assert 8 in pz_set and 24 in pz_set


def test_plan_skip_keys_excludes_slice():
    dz = 32
    fg = np.zeros((dz, dz, dz), dtype=bool)
    fg[0, 8:14, 8:14] = True
    patch_size = (dz, dz, dz)
    padded_shape = (96, 96, 96)
    sz = sy = sx = slice(16, 16 + 32)
    seed_key = spatial_slices_to_tuple(sz, sy, sx)
    p0 = plan_border_expansion_centers_from_fg(
        fg, sz, sy, sx, patch_size, padded_shape, seed_key, max_centers=8, quant_vox=1
    )
    assert len(p0) == 1
    u, v, w = centered_spatial_slices_at_point(p0[0][0], p0[0][1], p0[0][2], patch_size, padded_shape)
    k0 = spatial_slices_to_tuple(u, v, w)
    p1 = plan_border_expansion_centers_from_fg(
        fg,
        sz,
        sy,
        sx,
        patch_size,
        padded_shape,
        seed_key,
        max_centers=8,
        quant_vox=1,
        skip_keys={k0},
    )
    assert p1 == []


def test_plan_interior_only_no_extras():
    dz = 16
    fg = np.zeros((dz, dz, dz), dtype=bool)
    fg[4:12, 4:12, 4:12] = True
    patch_size = (dz, dz, dz)
    padded_shape = (48, 48, 48)
    sz = sy = sx = slice(0, 16)
    seed_key = spatial_slices_to_tuple(sz, sy, sx)
    plans = plan_border_expansion_centers_from_fg(fg, sz, sy, sx, patch_size, padded_shape, seed_key, max_centers=8)
    assert plans == []


def test_plan_high_x_face_quant_vox_aligns_px():
    """Regression: px must use (px//qv)*qv, not *(px) which corrupts X expansion."""
    dz = dy = dx = 64
    fg = np.zeros((dz, dy, dx), dtype=bool)
    fg[30:36, 30:36, dx - 1] = True
    patch_size = (dz, dy, dx)
    padded_shape = (256, 256, 256)
    sz = sy = sx = slice(32, 32 + 64)
    seed_key = spatial_slices_to_tuple(sz, sy, sx)
    plans = plan_border_expansion_centers_from_fg(
        fg, sz, sy, sx, patch_size, padded_shape, seed_key, max_centers=4, quant_vox=8
    )
    assert len(plans) == 1
    pz, py, px = plans[0]
    assert px % 8 == 0
    u, v, w = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
    assert spatial_slices_to_tuple(u, v, w) != seed_key
