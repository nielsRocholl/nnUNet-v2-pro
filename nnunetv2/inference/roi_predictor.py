"""Prompt-aware single-patch inference: no full-volume sliding window on the predictor."""
import os
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import cc3d
import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import convert_preprocessed_to_original_space
from nnunetv2.utilities.inference_execution import inference_accumulator_dtype
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.inference.prompt_clustering import cluster_centroid_zyx
from nnunetv2.utilities.prompt_encoding import encode_points_to_heatmap_pair
from nnunetv2.utilities.roi_config import RoiPromptConfig


def points_dict_to_canonical(
    d: Dict[str, Any],
    points_space_override: Optional[str] = None,
) -> Tuple[List, str, Optional[str], str, Optional[int]]:
    """Parse points payload (same keys as points JSON file).

    Returns (points_raw, points_space, points_format, voxel_coordinate_frame, debug_patch_bbox_pad).
    """
    points = d.get("points")
    if points is None:
        raise KeyError("points payload must have 'points' key")
    points_space = points_space_override or d.get("points_space", "voxel")
    if points_space not in ("voxel", "world"):
        raise ValueError(f"points_space must be 'voxel' or 'world', got {points_space!r}")
    points_format = d.get("points_format")
    if points_format is None:
        points_format = "zyx_voxel" if points_space == "voxel" else "xyz_world"
    voxel_frame = d.get("voxel_coordinate_frame", "full")
    if voxel_frame not in ("full", "preprocessed"):
        raise ValueError(
            f"voxel_coordinate_frame must be 'full' or 'preprocessed', got {voxel_frame!r}"
        )
    raw_pad = d.get("debug_patch_bbox_pad")
    debug_patch_bbox_pad = int(raw_pad) if raw_pad is not None else None
    return points, points_space, points_format, voxel_frame, debug_patch_bbox_pad


def parse_points_json(
    path: str,
    points_space_override: Optional[str] = None,
) -> Tuple[List, str, Optional[str], str, Optional[int]]:
    """Load points from JSON file. See points_dict_to_canonical for return tuple."""
    return points_dict_to_canonical(load_json(path), points_space_override)


def map_points_zyx_unpadded_to_padded(
    points_zyx: List[Tuple[int, int, int]],
    slicer_revert: Tuple[slice, ...],
) -> List[Tuple[int, int, int]]:
    dz = slicer_revert[1].start
    dy = slicer_revert[2].start
    dx = slicer_revert[3].start
    return [(z + dz, y + dy, x + dx) for z, y, x in points_zyx]


def centered_spatial_slices_at_point(
    pz: int,
    py: int,
    px: int,
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    starts = []
    for p, ps, dim in zip((pz, py, px), patch_size, padded_shape):
        s = p - ps // 2
        s = max(0, min(s, dim - ps))
        starts.append(s)
    return tuple(
        slice(starts[i], starts[i] + patch_size[i]) for i in range(3)
    )


def spatial_slices_to_tuple(sz: slice, sy: slice, sx: slice) -> Tuple[int, int, int, int, int, int]:
    return (sz.start, sz.stop, sy.start, sy.stop, sx.start, sx.stop)


def shift_spatial_slices(
    sz: slice,
    sy: slice,
    sx: slice,
    axis: int,
    is_low_face: bool,
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    z0, y0, x0 = sz.start, sy.start, sx.start
    half = [patch_size[i] // 2 for i in range(3)]
    if axis == 0:
        z0 = z0 - half[0] if is_low_face else z0 + half[0]
    elif axis == 1:
        y0 = y0 - half[1] if is_low_face else y0 + half[1]
    else:
        x0 = x0 - half[2] if is_low_face else x0 + half[2]
    z0 = max(0, min(z0, padded_shape[0] - patch_size[0]))
    y0 = max(0, min(y0, padded_shape[1] - patch_size[1]))
    x0 = max(0, min(x0, padded_shape[2] - patch_size[2]))
    return (
        slice(z0, z0 + patch_size[0]),
        slice(y0, y0 + patch_size[1]),
        slice(x0, x0 + patch_size[2]),
    )


def local_prompt_points_for_patch(
    seed_padded: Optional[Tuple[int, int, int]],
    sz: slice,
    sy: slice,
    sx: slice,
    patch_size: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    if seed_padded is None:
        return []
    pz, py, px = seed_padded
    if sz.start <= pz < sz.stop and sy.start <= py < sy.stop and sx.start <= px < sx.stop:
        return [(pz - sz.start, py - sy.start, px - sx.start)]
    return [(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)]


def local_points_padded_in_tile(
    points_padded: List[Tuple[int, int, int]],
    sz: slice,
    sy: slice,
    sx: slice,
) -> List[Tuple[int, int, int]]:
    o: List[Tuple[int, int, int]] = []
    for pz, py, px in points_padded:
        if sz.start <= pz < sz.stop and sy.start <= py < sy.stop and sx.start <= px < sx.stop:
            o.append((pz - sz.start, py - sy.start, px - sx.start))
    return o


def _maybe_to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t if t.device == device else t.to(device)


@torch.inference_mode()
def _patch_fg_tensor_from_logits(patch_logits: torch.Tensor, label_manager) -> torch.Tensor:
    """Foreground bool (D,H,W) on same device as logits; avoids moving full C×D×H×W logits to CPU."""
    x = patch_logits.float()
    if label_manager.has_regions:
        probs = label_manager.apply_inference_nonlin(x)
        seg = torch.zeros(probs.shape[1:], dtype=torch.int16, device=patch_logits.device)
        for i, c in enumerate(label_manager.regions_class_order):
            seg[probs[i] > 0.5] = c
        return seg > 0
    seg = x.argmax(0)
    fg = torch.zeros_like(seg, dtype=torch.bool)
    for fl in label_manager.foreground_labels:
        fg |= seg == int(fl)
    return fg


def touching_patch_faces_from_logits(
    patch_logits: torch.Tensor,
    label_manager,
) -> List[Tuple[int, bool]]:
    """Return list of (axis, is_low_face) for patch faces touched by foreground."""
    fg = _patch_fg_bool_from_logits(patch_logits, label_manager)
    if not fg.any():
        return []
    dz, dy, dx = fg.shape
    faces: List[Tuple[int, bool]] = []
    if fg[0].any():
        faces.append((0, True))
    if fg[dz - 1].any():
        faces.append((0, False))
    if fg[:, 0].any():
        faces.append((1, True))
    if fg[:, dy - 1].any():
        faces.append((1, False))
    if fg[:, :, 0].any():
        faces.append((2, True))
    if fg[:, :, dx - 1].any():
        faces.append((2, False))
    return faces


def _patch_fg_bool_from_logits(patch_logits: torch.Tensor, label_manager) -> np.ndarray:
    return _patch_fg_tensor_from_logits(patch_logits, label_manager).cpu().numpy()


def plan_border_expansion_centers_from_fg(
    fg: np.ndarray,
    sz: slice,
    sy: slice,
    sx: slice,
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
    seed_key: Tuple[int, int, int, int, int, int],
    max_centers: int,
    quant_vox: Optional[int] = None,
    skip_keys: Optional[Set[Tuple[int, int, int, int, int, int]]] = None,
) -> List[Tuple[int, int, int]]:
    if max_centers <= 0 or not fg.any():
        return []
    dz, dy, dx = fg.shape
    hull = np.zeros((dz, dy, dx), dtype=bool)
    hull[0] = hull[dz - 1] = True
    hull[:, 0] = hull[:, dy - 1] = True
    hull[:, :, 0] = hull[:, :, dx - 1] = True
    shell = fg & hull
    if not shell.any():
        return []
    lab = cc3d.connected_components(shell.astype(np.uint8), connectivity=26)
    n_lab = int(lab.max())
    if n_lab == 0:
        return []
    sizes = [(lab == i).sum() for i in range(1, n_lab + 1)]
    order = sorted(range(1, n_lab + 1), key=lambda i: sizes[i - 1], reverse=True)
    qv = quant_vox if quant_vox is not None else max(1, min(patch_size) // 8)
    skip = skip_keys if skip_keys is not None else set()
    seen: set = set()
    out: List[Tuple[int, int, int]] = []
    face_opts = [(0, True), (0, False), (1, True), (1, False), (2, True), (2, False)]
    for li in order:
        if len(out) >= max_centers:
            break
        zz_i, yy_i, xx_i = np.where(lab == li)
        if zz_i.size == 0:
            continue
        mz, my, mx = float(zz_i.mean()), float(yy_i.mean()), float(xx_i.mean())
        c_fac = [
            np.sum(zz_i == 0),
            np.sum(zz_i == dz - 1),
            np.sum(yy_i == 0),
            np.sum(yy_i == dy - 1),
            np.sum(xx_i == 0),
            np.sum(xx_i == dx - 1),
        ]
        face_indices = [i for i in range(6) if c_fac[i] > 0]
        face_indices.sort(key=lambda i: c_fac[i], reverse=True)
        for fi in face_indices:
            if len(out) >= max_centers:
                break
            axis, is_low = face_opts[fi]
            nsz, nsy, nsx = shift_spatial_slices(sz, sy, sx, axis, is_low, patch_size, padded_shape)
            Z0 = int(np.clip(sz.start + int(round(mz)), 0, padded_shape[0] - 1))
            Y0 = int(np.clip(sy.start + int(round(my)), 0, padded_shape[1] - 1))
            X0 = int(np.clip(sx.start + int(round(mx)), 0, padded_shape[2] - 1))
            if axis == 0:
                pz = int(np.clip(nsz.start + patch_size[0] // 2, 0, padded_shape[0] - 1))
                py, px = Y0, X0
            elif axis == 1:
                pz, py, px = Z0, int(np.clip(nsy.start + patch_size[1] // 2, 0, padded_shape[1] - 1)), X0
            else:
                pz, py, px = Z0, Y0, int(np.clip(nsx.start + patch_size[2] // 2, 0, padded_shape[2] - 1))
            if qv > 1:
                pz = (pz // qv) * qv
                py = (py // qv) * qv
                px = (px // qv) * qv
            cz, cy, cx = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
            s2 = spatial_slices_to_tuple(cz, cy, cx)
            if s2 == seed_key or s2 in seen or s2 in skip:
                continue
            seen.add(s2)
            out.append((pz, py, px))
    return out


def plan_border_expansion_centers_from_logits(
    patch_logits: torch.Tensor,
    label_manager,
    sz: slice,
    sy: slice,
    sx: slice,
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
    seed_key: Tuple[int, int, int, int, int, int],
    max_centers: int,
    quant_vox: Optional[int] = None,
    skip_keys: Optional[Set[Tuple[int, int, int, int, int, int]]] = None,
) -> List[Tuple[int, int, int]]:
    fg = _patch_fg_bool_from_logits(patch_logits, label_manager)
    return plan_border_expansion_centers_from_fg(
        fg, sz, sy, sx, patch_size, padded_shape, seed_key, max_centers, quant_vox, skip_keys
    )


def background_logits_vector(
    label_manager,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if label_manager.has_regions:
        return torch.full((num_heads,), -10.0, device=device, dtype=dtype)
    v = torch.full((num_heads,), -10.0, device=device, dtype=dtype)
    v[0] = 10.0
    return v


def safe_divide_merged_logits(
    predicted_logits: torch.Tensor,
    n_predictions: torch.Tensor,
    background_vec: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """In-place: divide where n_predictions > eps; else set logits to background_vec per voxel."""
    valid = n_predictions > eps
    inv = torch.clamp(n_predictions, min=eps)
    scaled = predicted_logits / inv
    bg_exp = background_vec.view(-1, 1, 1, 1).expand_as(predicted_logits)
    valid_exp = valid.unsqueeze(0).expand_as(predicted_logits)
    predicted_logits.copy_(torch.where(valid_exp, scaled, bg_exp))


def _write_3d_patch_nifti(vol_zyx: np.ndarray, out_fspath: str, affine: np.ndarray) -> None:
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))
    nib.save(nib.Nifti1Image(vol_xyz.astype(np.float32, copy=False), affine), out_fspath)


def _viewer_rescale_normalized_patch(vol_zyx: np.ndarray) -> np.ndarray:
    """nnU-Net uses masked Z-score style values, not HU; stretch for typical CT W/L sliders."""
    v = np.asarray(vol_zyx, dtype=np.float64)
    m = np.isfinite(v)
    if not np.any(m):
        return np.zeros_like(v, dtype=np.float32)
    lo, hi = np.percentile(v[m], [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    out = (v - lo) / (hi - lo) * 2000.0 - 1000.0
    return np.clip(out, -1024.0, 3071.0).astype(np.float32)


def _bbox_from_embedded_patch(
    orig_zyx: np.ndarray, pad_vox: Optional[int] = None
) -> Tuple[int, int, int, int, int, int]:
    if pad_vox is None:
        pad_vox = int(os.environ.get("NNUNET_DEBUG_PATCH_BBOX_PAD", "32"))
    m = np.abs(orig_zyx) > 1e-5
    if not np.any(m):
        m = orig_zyx != 0
    if not np.any(m):
        nz, ny, nx = orig_zyx.shape
        return 0, nz, 0, ny, 0, nx
    zz, yy, xx = np.where(m)
    nz, ny, nx = orig_zyx.shape
    z0 = max(0, int(zz.min()) - pad_vox)
    z1 = min(nz, int(zz.max()) + 1 + pad_vox)
    y0 = max(0, int(yy.min()) - pad_vox)
    y1 = min(ny, int(yy.max()) + 1 + pad_vox)
    x0 = max(0, int(xx.min()) - pad_vox)
    x1 = min(nx, int(xx.max()) + 1 + pad_vox)
    return z0, z1, y0, y1, x0, x1


def _write_sitk_zyx_crop(properties_dict: dict, full_zyx: np.ndarray, crop_zyx: np.ndarray, z0: int, y0: int, x0: int, path: str) -> None:
    import SimpleITK as sitk

    sp = properties_dict["sitk_stuff"]["spacing"]
    org = properties_dict["sitk_stuff"]["origin"]
    direc = properties_dict["sitk_stuff"]["direction"]
    full_itk = sitk.GetImageFromArray(full_zyx)
    full_itk.SetSpacing(sp)
    full_itk.SetOrigin(org)
    full_itk.SetDirection(direc)
    itk_idx = (int(x0), int(y0), int(z0))
    o_new = full_itk.TransformIndexToPhysicalPoint(itk_idx)
    sub_itk = sitk.GetImageFromArray(crop_zyx)
    sub_itk.SetSpacing(full_itk.GetSpacing())
    sub_itk.SetDirection(full_itk.GetDirection())
    sub_itk.SetOrigin(o_new)
    sitk.WriteImage(sub_itk, path, True)


def _single_channel_patch_to_native_nifti(
    patch_zyx: np.ndarray,
    padded_shape: Tuple[int, int, int],
    sz: slice,
    sy: slice,
    sx: slice,
    slicer_revert: Tuple,
    properties: dict,
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    out_path: str,
    is_seg: bool,
    viewer_rescale: bool,
    debug_bbox_pad_vox: Optional[int] = None,
) -> None:
    canvas = np.zeros((1,) + tuple(padded_shape), dtype=np.float32)
    canvas[0, sz, sy, sx] = np.asarray(patch_zyx, dtype=np.float32)
    canvas_u = canvas[(slice(None), *slicer_revert[1:])]
    vol_prep = canvas_u[0]
    full_orig = convert_preprocessed_to_original_space(
        vol_prep, properties, plans_manager, configuration_manager, is_seg=is_seg
    )
    z0, z1, y0, y1, x0, x1 = _bbox_from_embedded_patch(
        full_orig, pad_vox=debug_bbox_pad_vox
    )
    crop_raw = full_orig[z0:z1, y0:y1, x0:x1]
    # Bbox must be on raw converted grid: global viewer rescale fills/clips background and masks no longer vanish.
    crop = (
        _viewer_rescale_normalized_patch(crop_raw) if viewer_rescale else crop_raw
    )
    _write_sitk_zyx_crop(properties, full_orig, crop, z0, y0, x0, out_path)


def save_single_patch_debug_niftis(
    image_c_zyx: torch.Tensor,
    prompt_2_zyx: torch.Tensor,
    out_path: str,
    spacing_zyx: Tuple[float, float, float],
    *,
    save_prompt_channels: bool,
    native_geometry: Optional[Dict[str, Any]] = None,
) -> None:
    """Save exact network input (*_preprocessed*) and a *_viewer* volume (pseudo-HU spread for CT viewers).

    If native_geometry is set (padded_shape, slices, slicer_revert, properties, plans_manager,
    configuration_manager), volumes are embedded in full preprocessed space, inverted with
    convert_preprocessed_to_original_space, and written with SimpleITK using sitk_stuff so viewers
    match exported predictions. Otherwise fallback: diagonal affine + nibabel (legacy).
    """
    if image_c_zyx.ndim != 4:
        raise ValueError("image_c_zyx must be (C, Z, Y, X)")
    if prompt_2_zyx.shape[0] != 2:
        raise ValueError("prompt must be (2, Z, Y, X)")
    szf, syf, sxf = float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2])
    affine = np.diag([szf, syf, sxf, 1.0]).astype(np.float64)

    base = out_path
    if base.endswith(".nii.gz"):
        base = base[: -len(".nii.gz")]
    elif base.endswith(".nii"):
        base = base[: -len(".nii")]
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        maybe_mkdir_p(d)

    img = image_c_zyx.detach().cpu().float().numpy()
    n_img = img.shape[0]
    written: List[str] = []

    use_native = (
        native_geometry is not None
        and native_geometry.get("properties", {}).get("sitk_stuff") is not None
    )
    if use_native:
        props = native_geometry["properties"]
        pm: PlansManager = native_geometry["plans_manager"]
        cm: ConfigurationManager = native_geometry["configuration_manager"]
        padded_shape: Tuple[int, int, int] = tuple(native_geometry["padded_shape"])
        slz: slice = native_geometry["sz"]
        sly: slice = native_geometry["sy"]
        slx: slice = native_geometry["sx"]
        sr = native_geometry["slicer_revert"]
        bbox_pad: Optional[int] = native_geometry.get("debug_bbox_pad_vox")
        if n_img == 1:
            p = f"{base}_preprocessed.nii.gz"
            _single_channel_patch_to_native_nifti(
                img[0],
                padded_shape,
                slz,
                sly,
                slx,
                sr,
                props,
                pm,
                cm,
                p,
                is_seg=False,
                viewer_rescale=False,
                debug_bbox_pad_vox=bbox_pad,
            )
            written.append(os.path.basename(p))
            pv = f"{base}_viewer.nii.gz"
            _single_channel_patch_to_native_nifti(
                img[0],
                padded_shape,
                slz,
                sly,
                slx,
                sr,
                props,
                pm,
                cm,
                pv,
                is_seg=False,
                viewer_rescale=True,
                debug_bbox_pad_vox=bbox_pad,
            )
            written.append(os.path.basename(pv))
        else:
            for c in range(n_img):
                p = f"{base}_preprocessed_ch{c:02d}.nii.gz"
                _single_channel_patch_to_native_nifti(
                    img[c],
                    padded_shape,
                    slz,
                    sly,
                    slx,
                    sr,
                    props,
                    pm,
                    cm,
                    p,
                    is_seg=False,
                    viewer_rescale=False,
                    debug_bbox_pad_vox=bbox_pad,
                )
                written.append(os.path.basename(p))
                pv = f"{base}_viewer_ch{c:02d}.nii.gz"
                _single_channel_patch_to_native_nifti(
                    img[c],
                    padded_shape,
                    slz,
                    sly,
                    slx,
                    sr,
                    props,
                    pm,
                    cm,
                    pv,
                    is_seg=False,
                    viewer_rescale=True,
                    debug_bbox_pad_vox=bbox_pad,
                )
                written.append(os.path.basename(pv))
        if save_prompt_channels:
            pr = prompt_2_zyx.detach().cpu().float().numpy()
            for i in range(2):
                p = f"{base}_prompt_{i}.nii.gz"
                _single_channel_patch_to_native_nifti(
                    pr[i],
                    padded_shape,
                    slz,
                    sly,
                    slx,
                    sr,
                    props,
                    pm,
                    cm,
                    p,
                    is_seg=False,
                    viewer_rescale=False,
                    debug_bbox_pad_vox=bbox_pad,
                )
                written.append(os.path.basename(p))
    else:
        if n_img == 1:
            p = f"{base}_preprocessed.nii.gz"
            _write_3d_patch_nifti(img[0], p, affine)
            written.append(os.path.basename(p))
            pv = f"{base}_viewer.nii.gz"
            _write_3d_patch_nifti(_viewer_rescale_normalized_patch(img[0]), pv, affine)
            written.append(os.path.basename(pv))
        else:
            for c in range(n_img):
                p = f"{base}_preprocessed_ch{c:02d}.nii.gz"
                _write_3d_patch_nifti(img[c], p, affine)
                written.append(os.path.basename(p))
                pv = f"{base}_viewer_ch{c:02d}.nii.gz"
                _write_3d_patch_nifti(_viewer_rescale_normalized_patch(img[c]), pv, affine)
                written.append(os.path.basename(pv))

        if save_prompt_channels:
            pr = prompt_2_zyx.detach().cpu().float().numpy()
            for i in range(2):
                p = f"{base}_prompt_{i}.nii.gz"
                _write_3d_patch_nifti(pr[i], p, affine)
                written.append(os.path.basename(p))

    note = (
        "native orientation (sitk_stuff + convert_preprocessed_to_original_space)"
        if use_native
        else "diagonal affine fallback"
    )
    print(
        f"[single_patch] saved debug NIfTI: {', '.join(written)} "
        f"({note}; viewer ≈ stretched for CT W/L)",
        flush=True,
    )


class nnUNetROIPredictor(nnUNetPredictor):
    """Prompt-aware inference: use predict_logits_single_patch, not full-volume sliding."""

    @torch.inference_mode()
    def predict_logits_single_patch(
        self,
        data: torch.Tensor,
        points_zyx: List[Tuple[int, int, int]],
        cfg: RoiPromptConfig,
        *,
        encode_prompt: bool = False,
        save_debug_patch: Optional[str] = None,
        debug_patch_spacing_zyx: Optional[Tuple[float, float, float]] = None,
        save_debug_patch_prompts: bool = False,
        debug_native_geometry: Optional[Dict[str, Any]] = None,
        border_expand: bool = False,
        max_border_expand_extra: int = 16,
    ) -> torch.Tensor:
        """One tile centered on the first seed; optional patch-local prompt heatmap (no dilated ROI sliding)."""
        from torch._dynamo import OptimizedModule

        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None
        for fi, params in enumerate(self.list_of_parameters):
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
            fold_logits = self._predict_logits_single_patch_single_fold(
                data,
                points_zyx,
                cfg,
                encode_prompt,
                save_debug_patch=save_debug_patch if fi == 0 else None,
                debug_patch_spacing_zyx=debug_patch_spacing_zyx,
                save_debug_patch_prompts=save_debug_patch_prompts,
                debug_native_geometry=debug_native_geometry if fi == 0 else None,
                border_expand=border_expand,
                max_border_expand_extra=max_border_expand_extra,
                report_border_tile_progress=(fi == 0),
            )
            if prediction is None:
                prediction = fold_logits
            else:
                prediction = prediction + fold_logits
        if len(self.list_of_parameters) > 1:
            prediction = prediction / len(self.list_of_parameters)
        torch.set_num_threads(n_threads)
        return prediction

    @torch.inference_mode()
    def predict_logits_from_prompt_clusters(
        self,
        data: torch.Tensor,
        clusters: List[List[Tuple[int, int, int]]],
        cfg: RoiPromptConfig,
        *,
        encode_prompt: bool = True,
        encode_per_cluster: Optional[List[bool]] = None,
        save_debug_patch: Optional[str] = None,
        debug_patch_spacing_zyx: Optional[Tuple[float, float, float]] = None,
        save_debug_patch_prompts: bool = False,
        debug_native_geometry: Optional[Dict[str, Any]] = None,
        border_expand: bool = False,
        max_border_expand_extra: int = 16,
        cross_cluster_neg: bool = False,
    ) -> torch.Tensor:
        """Multi-cluster adaptive tiling: points are pre-z-clustered in preprocessed (unpadded) zyx. One seed+BFS per cluster, merged with Gaussian weights."""
        from torch._dynamo import OptimizedModule

        if not clusters or any(len(c) == 0 for c in clusters):
            raise ValueError("predict_logits_from_prompt_clusters requires non-empty clusters with at least one point each")
        if encode_per_cluster is not None and len(encode_per_cluster) != len(clusters):
            raise ValueError("encode_per_cluster length must match number of clusters")
        if cross_cluster_neg and encode_per_cluster is not None and not all(encode_per_cluster):
            raise ValueError("cross_cluster_neg requires prompt encoding on every cluster; gate left some clusters probe-only")
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None
        for fi, params in enumerate(self.list_of_parameters):
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
            fold_logits = self._predict_logits_multi_cluster_single_fold(
                data,
                clusters,
                cfg,
                encode_prompt,
                encode_per_cluster=encode_per_cluster,
                save_debug_patch=save_debug_patch if fi == 0 else None,
                debug_patch_spacing_zyx=debug_patch_spacing_zyx,
                save_debug_patch_prompts=save_debug_patch_prompts,
                debug_native_geometry=debug_native_geometry if fi == 0 else None,
                border_expand=border_expand,
                max_border_expand_extra=max_border_expand_extra,
                cross_cluster_neg=cross_cluster_neg,
                report_border_tile_progress=(fi == 0),
            )
            if prediction is None:
                prediction = fold_logits
            else:
                prediction = prediction + fold_logits
        if len(self.list_of_parameters) > 1:
            prediction = prediction / len(self.list_of_parameters)
        torch.set_num_threads(n_threads)
        return prediction

    def _predict_logits_multi_cluster_single_fold(
        self,
        data: torch.Tensor,
        clusters: List[List[Tuple[int, int, int]]],
        cfg: RoiPromptConfig,
        encode_prompt: bool,
        encode_per_cluster: Optional[List[bool]] = None,
        save_debug_patch: Optional[str] = None,
        debug_patch_spacing_zyx: Optional[Tuple[float, float, float]] = None,
        save_debug_patch_prompts: bool = False,
        debug_native_geometry: Optional[Dict[str, Any]] = None,
        border_expand: bool = False,
        max_border_expand_extra: int = 16,
        cross_cluster_neg: bool = False,
        report_border_tile_progress: bool = True,
    ) -> torch.Tensor:
        self.network = self.network.to(self.device)
        self.network.eval()
        patch_size = tuple(self.configuration_manager.patch_size)
        prompt_cfg = cfg.prompt
        num_heads = self.label_manager.num_segmentation_heads
        data_f = data.float()
        data_padded, slicer_revert = pad_nd_image(
            data_f, patch_size, "constant", {"value": 0}, True, None
        )
        padded_shape = tuple(data_padded.shape[1:])
        data_padded = data_padded.to(self.device)
        results_device = (
            self.device if self.perform_everything_on_device and self.device.type != "cpu" else torch.device("cpu")
        )
        acc_dtype = inference_accumulator_dtype(results_device)
        if self.use_gaussian:
            gaussian = compute_gaussian(
                tuple(patch_size),
                sigma_scale=1.0 / 8,
                value_scaling_factor=10,
                dtype=torch.float32,
                device=results_device,
            )
        else:
            gaussian = torch.ones(patch_size, dtype=torch.float32, device=results_device)
        gaussian_acc = gaussian.to(dtype=acc_dtype)
        bg_vec = background_logits_vector(self.label_manager, num_heads, results_device, dtype=acc_dtype)

        clusters_pad: List[List[Tuple[int, int, int]]] = [
            map_points_zyx_unpadded_to_padded(pts, slicer_revert) for pts in clusters
        ]
        centroids_pad = [cluster_centroid_zyx(cp) for cp in clusters_pad]

        n_img_ch = int(data_padded.shape[0])
        workon = torch.empty((1, n_img_ch + 2, *patch_size), device=self.device, dtype=torch.float32)

        global_forwarded: Set[Tuple[int, int, int, int, int, int]] = set()
        first_tile_debug_done = False

        def fill_tile(sz: slice, sy: slice, sx: slice, cl_idx: int) -> bool:
            nonlocal first_tile_debug_done
            k = spatial_slices_to_tuple(sz, sy, sx)
            if k in global_forwarded:
                warnings.warn(
                    f"multi-prompt: skipping duplicate tile {k!r} (first forward kept)",
                    UserWarning,
                    stacklevel=3,
                )
                return False
            global_forwarded.add(k)
            ep = encode_prompt if encode_per_cluster is None else bool(encode_per_cluster[cl_idx])
            pts = clusters_pad[cl_idx]
            pos = local_points_padded_in_tile(pts, sz, sy, sx)
            if cross_cluster_neg and ep:
                others = [centroids_pad[i] for i in range(len(centroids_pad)) if i != cl_idx]
                neg = local_points_padded_in_tile(others, sz, sy, sx)
            else:
                neg = []
            self._fill_workon_patch(
                workon, data_padded, sz, sy, sx, n_img_ch, ep, pos, neg, prompt_cfg, patch_size
            )
            if save_debug_patch and (not first_tile_debug_done):
                sp = debug_patch_spacing_zyx if debug_patch_spacing_zyx is not None else (1.0, 1.0, 1.0)
                ng = None
                if debug_native_geometry is not None:
                    ng = {
                        **debug_native_geometry,
                        "padded_shape": padded_shape,
                        "sz": sz,
                        "sy": sy,
                        "sx": sx,
                        "slicer_revert": slicer_revert,
                    }
                save_single_patch_debug_niftis(
                    workon[0, :n_img_ch].clone(),
                    workon[0, n_img_ch:].clone(),
                    save_debug_patch,
                    sp,
                    save_prompt_channels=save_debug_patch_prompts,
                    native_geometry=ng,
                )
                first_tile_debug_done = True
            return True

        try:
            if not border_expand:
                bg_plane = bg_vec.view(-1, 1, 1, 1).to(dtype=acc_dtype)
                predicted_logits = bg_plane.expand(num_heads, *padded_shape).clone()
                for cl_idx in range(len(clusters_pad)):
                    cz, cy, cx = centroids_pad[cl_idx]
                    sz, sy, sx = centered_spatial_slices_at_point(cz, cy, cx, patch_size, padded_shape)
                    if not fill_tile(sz, sy, sx, cl_idx):
                        if self.verbose:
                            print(
                                f"[from_prompts] cluster {cl_idx}: seed tile duplicate, skipped",
                                flush=True,
                            )
                        continue
                    pred_raw = _maybe_to_device(self._internal_maybe_mirror_and_predict(workon)[0], results_device)
                    merged = pred_raw.to(dtype=acc_dtype)
                    predicted_logits[(slice(None), sz, sy, sx)] = merged
                predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
                if torch.any(torch.isinf(predicted_logits)):
                    raise RuntimeError("Encountered inf in predicted array. Aborting.")
                return predicted_logits.float().cpu()

            predicted_logits = torch.zeros((num_heads, *padded_shape), dtype=acc_dtype, device=results_device)
            n_predictions = torch.zeros(padded_shape, dtype=acc_dtype, device=results_device)
            bfcb = getattr(self, "_border_tile_progress_cb", None)
            total_bt = 1 + max_border_expand_extra
            tile_count = [0]

            for cl_idx in range(len(clusters_pad)):
                cz, cy, cx = centroids_pad[cl_idx]
                sz, sy, sx = centered_spatial_slices_at_point(cz, cy, cx, patch_size, padded_shape)
                if not fill_tile(sz, sy, sx, cl_idx):
                    if self.verbose:
                        print(
                            f"[from_prompts] cluster {cl_idx}: seed tile duplicate, skipped",
                            flush=True,
                        )
                    continue
                pred_raw = _maybe_to_device(self._internal_maybe_mirror_and_predict(workon)[0], results_device)
                acc = (pred_raw * gaussian if self.use_gaussian else pred_raw).to(dtype=acc_dtype)
                predicted_logits[(slice(None), sz, sy, sx)] += acc
                n_predictions[sz, sy, sx] += gaussian_acc
                if report_border_tile_progress and bfcb is not None:
                    tile_count[0] += 1
                    bfcb(tile_count[0], total_bt * max(1, len(clusters_pad)))
                seed_key = spatial_slices_to_tuple(sz, sy, sx)
                visited_keys: Set[Tuple[int, int, int, int, int, int]] = {seed_key}
                q: deque = deque(
                    plan_border_expansion_centers_from_logits(
                        pred_raw,
                        self.label_manager,
                        sz,
                        sy,
                        sx,
                        patch_size,
                        padded_shape,
                        seed_key,
                        max_border_expand_extra,
                        skip_keys=visited_keys,
                    )
                )
                tiles_done = 0
                while q and tiles_done < max_border_expand_extra:
                    pz_e, py_e, px_e = q.popleft()
                    sz_e, sy_e, sx_e = centered_spatial_slices_at_point(
                        pz_e, py_e, px_e, patch_size, padded_shape
                    )
                    k_e = spatial_slices_to_tuple(sz_e, sy_e, sx_e)
                    if k_e in visited_keys:
                        continue
                    visited_keys.add(k_e)
                    if not fill_tile(sz_e, sy_e, sx_e, cl_idx):
                        continue
                    raw_e = _maybe_to_device(self._internal_maybe_mirror_and_predict(workon)[0], results_device)
                    pred_e = (raw_e * gaussian if self.use_gaussian else raw_e).to(dtype=acc_dtype)
                    predicted_logits[(slice(None), sz_e, sy_e, sx_e)] += pred_e
                    n_predictions[sz_e, sy_e, sx_e] += gaussian_acc
                    tiles_done += 1
                    if report_border_tile_progress and bfcb is not None:
                        tile_count[0] += 1
                        bfcb(tile_count[0], total_bt * max(1, len(clusters_pad)))
                    budget = max_border_expand_extra - tiles_done
                    if budget > 0:
                        for c in plan_border_expansion_centers_from_logits(
                            raw_e,
                            self.label_manager,
                            sz_e,
                            sy_e,
                            sx_e,
                            patch_size,
                            padded_shape,
                            seed_key,
                            budget,
                            skip_keys=visited_keys,
                        ):
                            q.append(c)
                if self.verbose and tiles_done:
                    print(
                        f"[from_prompts] border_expand cluster {cl_idx}: {tiles_done} extra tile(s)",
                        flush=True,
                    )
            safe_divide_merged_logits(predicted_logits, n_predictions, bg_vec)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError("Encountered inf in predicted array. Aborting.")
        except Exception as e:
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits.cpu().float()

    def _fill_workon_patch(
        self,
        workon: torch.Tensor,
        data_padded: torch.Tensor,
        sz: slice,
        sy: slice,
        sx: slice,
        n_img_ch: int,
        encode_prompt: bool,
        pos_local: List[Tuple[int, int, int]],
        neg_local: List[Tuple[int, int, int]],
        prompt_cfg,
        patch_size: Tuple[int, int, int],
    ) -> None:
        sl_img = (slice(None), sz, sy, sx)
        workon[0, :n_img_ch].copy_(data_padded[sl_img])
        if not encode_prompt:
            workon[0, n_img_ch:].zero_()
            return
        pr = encode_points_to_heatmap_pair(
            pos_local,
            neg_local,
            patch_size,
            prompt_cfg.point_radius_vox,
            prompt_cfg.encoding,
            device=self.device,
            intensity_scale=prompt_cfg.prompt_intensity_scale,
        )
        workon[0, n_img_ch:].copy_(pr.float())

    def _predict_logits_single_patch_single_fold(
        self,
        data: torch.Tensor,
        points_zyx: List[Tuple[int, int, int]],
        cfg: RoiPromptConfig,
        encode_prompt: bool,
        save_debug_patch: Optional[str] = None,
        debug_patch_spacing_zyx: Optional[Tuple[float, float, float]] = None,
        save_debug_patch_prompts: bool = False,
        debug_native_geometry: Optional[Dict[str, Any]] = None,
        border_expand: bool = False,
        max_border_expand_extra: int = 16,
        report_border_tile_progress: bool = True,
    ) -> torch.Tensor:
        if not points_zyx:
            raise ValueError(
                "predict_logits_single_patch requires at least one point in preprocessed zyx voxel space"
            )
        self.network = self.network.to(self.device)
        self.network.eval()
        patch_size = tuple(self.configuration_manager.patch_size)
        prompt_cfg = cfg.prompt
        num_heads = self.label_manager.num_segmentation_heads
        data_f = data.float()
        data_padded, slicer_revert = pad_nd_image(
            data_f, patch_size, "constant", {"value": 0}, True, None
        )
        padded_shape = tuple(data_padded.shape[1:])
        data_padded = data_padded.to(self.device)
        results_device = (
            self.device if self.perform_everything_on_device and self.device.type != "cpu" else torch.device("cpu")
        )
        acc_dtype = inference_accumulator_dtype(results_device)
        if self.use_gaussian:
            gaussian = compute_gaussian(
                tuple(patch_size),
                sigma_scale=1.0 / 8,
                value_scaling_factor=10,
                dtype=torch.float32,
                device=results_device,
            )
        else:
            gaussian = torch.ones(patch_size, dtype=torch.float32, device=results_device)
        gaussian_acc = gaussian.to(dtype=acc_dtype)

        points_pad = map_points_zyx_unpadded_to_padded(points_zyx, slicer_revert)
        pz, py, px = points_pad[0]
        seed_pad: Tuple[int, int, int] = (pz, py, px)
        sz, sy, sx = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
        bg_vec = background_logits_vector(self.label_manager, num_heads, results_device, dtype=acc_dtype)

        n_img_ch = int(data_padded.shape[0])
        workon = torch.empty((1, n_img_ch + 2, *patch_size), device=self.device, dtype=torch.float32)

        try:
            pos0 = local_prompt_points_for_patch(seed_pad, sz, sy, sx, patch_size)
            self._fill_workon_patch(
                workon,
                data_padded,
                sz,
                sy,
                sx,
                n_img_ch,
                encode_prompt,
                pos0,
                [],
                prompt_cfg,
                patch_size,
            )
            if save_debug_patch:
                sp = debug_patch_spacing_zyx if debug_patch_spacing_zyx is not None else (1.0, 1.0, 1.0)
                ng = None
                if debug_native_geometry is not None:
                    ng = {
                        **debug_native_geometry,
                        "padded_shape": padded_shape,
                        "sz": sz,
                        "sy": sy,
                        "sx": sx,
                        "slicer_revert": slicer_revert,
                    }
                save_single_patch_debug_niftis(
                    workon[0, :n_img_ch].clone(),
                    workon[0, n_img_ch:].clone(),
                    save_debug_patch,
                    sp,
                    save_prompt_channels=save_debug_patch_prompts,
                    native_geometry=ng,
                )
            pred_raw = _maybe_to_device(self._internal_maybe_mirror_and_predict(workon)[0], results_device)

            if not border_expand:
                merged = pred_raw.to(dtype=acc_dtype)
                bg_plane = bg_vec.view(-1, 1, 1, 1).to(dtype=acc_dtype)
                predicted_logits = bg_plane.expand(num_heads, *padded_shape).clone()
                predicted_logits[:, sz, sy, sx] = merged
                predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
                if torch.any(torch.isinf(predicted_logits)):
                    raise RuntimeError("Encountered inf in predicted array. Aborting.")
                return predicted_logits.float().cpu()

            predicted_logits = torch.zeros((num_heads, *padded_shape), dtype=acc_dtype, device=results_device)
            n_predictions = torch.zeros(padded_shape, dtype=acc_dtype, device=results_device)
            pred_acc = (pred_raw * gaussian if self.use_gaussian else pred_raw).to(dtype=acc_dtype)
            predicted_logits[(slice(None), sz, sy, sx)] += pred_acc
            n_predictions[sz, sy, sx] += gaussian_acc

            seed_key = spatial_slices_to_tuple(sz, sy, sx)
            bfcb = getattr(self, "_border_tile_progress_cb", None)
            total_bt = 1 + max_border_expand_extra
            if report_border_tile_progress and border_expand and bfcb is not None:
                bfcb(1, total_bt)

            visited_keys: Set[Tuple[int, int, int, int, int, int]] = {seed_key}
            q: deque = deque(
                plan_border_expansion_centers_from_logits(
                    pred_raw,
                    self.label_manager,
                    sz,
                    sy,
                    sx,
                    patch_size,
                    padded_shape,
                    seed_key,
                    max_border_expand_extra,
                    skip_keys=visited_keys,
                )
            )
            tiles_done = 0
            # Batched forwards for queued tiles would require isolating mirroring TTA per batch element; not implemented.
            while q and tiles_done < max_border_expand_extra:
                pz_e, py_e, px_e = q.popleft()
                sz_e, sy_e, sx_e = centered_spatial_slices_at_point(
                    pz_e, py_e, px_e, patch_size, padded_shape
                )
                k_e = spatial_slices_to_tuple(sz_e, sy_e, sx_e)
                if k_e in visited_keys:
                    continue
                visited_keys.add(k_e)
                pos_e = local_prompt_points_for_patch(seed_pad, sz_e, sy_e, sx_e, patch_size)
                self._fill_workon_patch(
                    workon,
                    data_padded,
                    sz_e,
                    sy_e,
                    sx_e,
                    n_img_ch,
                    encode_prompt,
                    pos_e,
                    [],
                    prompt_cfg,
                    patch_size,
                )
                raw_e = _maybe_to_device(self._internal_maybe_mirror_and_predict(workon)[0], results_device)
                pred_e = (raw_e * gaussian if self.use_gaussian else raw_e).to(dtype=acc_dtype)
                predicted_logits[(slice(None), sz_e, sy_e, sx_e)] += pred_e
                n_predictions[sz_e, sy_e, sx_e] += gaussian_acc
                tiles_done += 1
                if report_border_tile_progress and bfcb is not None:
                    bfcb(1 + tiles_done, total_bt)
                budget = max_border_expand_extra - tiles_done
                if budget > 0:
                    for c in plan_border_expansion_centers_from_logits(
                        raw_e,
                        self.label_manager,
                        sz_e,
                        sy_e,
                        sx_e,
                        patch_size,
                        padded_shape,
                        seed_key,
                        budget,
                        skip_keys=visited_keys,
                    ):
                        q.append(c)
            if self.verbose and tiles_done:
                print(f"[single_patch] border_expand: {tiles_done} extra tile(s)", flush=True)

            safe_divide_merged_logits(predicted_logits, n_predictions, bg_vec)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError("Encountered inf in predicted array. Aborting.")
        except Exception as e:
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits.cpu().float()

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
        """Override to prevent accidental full-volume usage; use predict_logits_single_patch instead."""
        raise RuntimeError(
            "nnUNetROIPredictor does not support full-volume sliding. Use predict_logits_single_patch."
        )
