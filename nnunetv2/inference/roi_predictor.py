"""ROI-only inference: prompt-aware local sliding windows, no full-volume sliding."""
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import convert_preprocessed_to_original_space
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.prompt_encoding import encode_points_to_heatmap_pair
from nnunetv2.utilities.roi_config import RoiPromptConfig


def get_prompt_aware_slicers(
    image_size: Tuple[int, ...],
    patch_size: Tuple[int, int, int],
    tile_step_size: float,
    dense_prompt: Optional[torch.Tensor] = None,
) -> List[Tuple]:
    """Slicers for prompt-aware sliding: dilated bbox (prompt + patch_size/2, clamped).
    Single patch if dilated bbox < patch; else filter sliding windows.
    Empty prompt → 1 centered patch."""
    if len(patch_size) < len(image_size):
        raise NotImplementedError("ROI predictor supports 3D only")
    slicers = []
    if dense_prompt is None or dense_prompt.numel() == 0 or dense_prompt.max().item() <= 0:
        center = [image_size[i] // 2 for i in range(3)]
        start = [max(0, center[i] - patch_size[i] // 2) for i in range(3)]
        end = [min(image_size[i], start[i] + patch_size[i]) for i in range(3)]
        for i in range(3):
            if end[i] - start[i] < patch_size[i]:
                if end[i] == image_size[i]:
                    start[i] = max(0, end[i] - patch_size[i])
                else:
                    end[i] = min(image_size[i], start[i] + patch_size[i])
        slicers.append(tuple([slice(None)] + [slice(start[i], end[i]) for i in range(3)]))
        return slicers
    prompt_coords = torch.where(dense_prompt[0] > 0)
    if len(prompt_coords[0]) == 0:
        center = [image_size[i] // 2 for i in range(3)]
        start = [max(0, center[i] - patch_size[i] // 2) for i in range(3)]
        end = [min(image_size[i], start[i] + patch_size[i]) for i in range(3)]
        slicers.append(tuple([slice(None)] + [slice(start[i], end[i]) for i in range(3)]))
        return slicers
    prompt_min = [int(prompt_coords[i].min().item()) for i in range(3)]
    prompt_max = [int(prompt_coords[i].max().item()) for i in range(3)]
    half = [patch_size[i] // 2 for i in range(3)]
    d_min = [max(0, prompt_min[i] - half[i]) for i in range(3)]
    d_max = [min(image_size[i], prompt_max[i] + half[i]) for i in range(3)]
    dilated_extent = [d_max[i] - d_min[i] for i in range(3)]
    if all(dilated_extent[i] < patch_size[i] for i in range(3)):
        center = [(d_min[i] + d_max[i]) // 2 for i in range(3)]
        start = [max(0, center[i] - patch_size[i] // 2) for i in range(3)]
        end = [min(image_size[i], start[i] + patch_size[i]) for i in range(3)]
        for i in range(3):
            if end[i] - start[i] < patch_size[i]:
                if end[i] == image_size[i]:
                    start[i] = max(0, end[i] - patch_size[i])
                else:
                    end[i] = min(image_size[i], start[i] + patch_size[i])
        slicers.append(tuple([slice(None)] + [slice(start[i], end[i]) for i in range(3)]))
        return slicers
    steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                slc = tuple([
                    slice(None),
                    *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)],
                ])
                if (
                    d_min[0] < slc[1].stop and d_max[0] > slc[1].start
                    and d_min[1] < slc[2].stop and d_max[1] > slc[2].start
                    and d_min[2] < slc[3].stop and d_max[2] > slc[3].start
                ):
                    slicers.append(slc)
    return slicers


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


def touching_patch_faces_from_logits(
    patch_logits: torch.Tensor,
    label_manager,
) -> List[Tuple[int, bool]]:
    """Return list of (axis, is_low_face) for patch faces touched by foreground."""
    seg = label_manager.convert_logits_to_segmentation(patch_logits.float().cpu())
    if isinstance(seg, torch.Tensor):
        seg = seg.numpy()
    fg = np.zeros(seg.shape, dtype=bool)
    if label_manager.has_regions:
        fg = seg > 0
    else:
        for fl in label_manager.foreground_labels:
            fg |= seg == fl
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
    """ROI-only inference: never runs full-volume sliding."""

    @torch.inference_mode()
    def predict_logits_roi_mode(
        self,
        data: torch.Tensor,
        points_zyx: List[Tuple[int, int, int]],
        properties: dict,
        cfg: RoiPromptConfig,
        tile_step_size: Optional[float] = None,
    ) -> torch.Tensor:
        """Predict logits: dilated bbox sliding or per-point expanding patches."""
        from torch._dynamo import OptimizedModule

        tile_step_size = tile_step_size if tile_step_size is not None else self.tile_step_size
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        self._roi_patch_visits_capped = False
        prediction = None
        for params in self.list_of_parameters:
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
            if cfg.inference.roi_inference_mode == "per_point_patch":
                fold_logits = self._predict_logits_roi_per_point_patch_single_fold(data, points_zyx, cfg)
            else:
                fold_logits = self._predict_logits_roi_single_fold(
                    data, points_zyx, cfg, tile_step_size
                )
            if prediction is None:
                prediction = fold_logits
            else:
                prediction = prediction + fold_logits
        if len(self.list_of_parameters) > 1:
            prediction = prediction / len(self.list_of_parameters)
        torch.set_num_threads(n_threads)
        if getattr(self, "_roi_patch_visits_capped", False) and self.verbose:
            print(
                "[nnUNetROI] max_patch_expansion_visits reached; expansion stopped early for this case.",
                flush=True,
            )
        return prediction

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
            )
            if prediction is None:
                prediction = fold_logits
            else:
                prediction = prediction + fold_logits
        if len(self.list_of_parameters) > 1:
            prediction = prediction / len(self.list_of_parameters)
        torch.set_num_threads(n_threads)
        return prediction

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
        predicted_logits = torch.zeros(
            (num_heads, *padded_shape), dtype=torch.float32, device=results_device
        )
        n_predictions = torch.zeros(padded_shape, dtype=torch.float32, device=results_device)
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

        points_pad = map_points_zyx_unpadded_to_padded(points_zyx, slicer_revert)
        pz, py, px = points_pad[0]
        seed_pad: Tuple[int, int, int] = (pz, py, px)
        sz, sy, sx = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
        bg_vec = background_logits_vector(self.label_manager, num_heads, results_device)

        try:
            sl = (slice(None), sz, sy, sx)
            img_crop = data_padded[sl].clone().unsqueeze(0)
            if encode_prompt:
                local_pts = local_prompt_points_for_patch(seed_pad, sz, sy, sx, patch_size)
                prompt = encode_points_to_heatmap_pair(
                    local_pts,
                    [],
                    patch_size,
                    prompt_cfg.point_radius_vox,
                    prompt_cfg.encoding,
                    device=self.device,
                    intensity_scale=prompt_cfg.prompt_intensity_scale,
                ).to(dtype=img_crop.dtype)
            else:
                prompt = torch.zeros(2, *patch_size, dtype=img_crop.dtype, device=self.device)
            workon = torch.cat([img_crop[0], prompt], dim=0).unsqueeze(0).contiguous()
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
                    img_crop[0],
                    prompt,
                    save_debug_patch,
                    sp,
                    save_prompt_channels=save_debug_patch_prompts,
                    native_geometry=ng,
                )
            pred_raw = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
            pred = pred_raw * gaussian if self.use_gaussian else pred_raw
            predicted_logits[(slice(None), sz, sy, sx)] += pred
            n_predictions[sz, sy, sx] += gaussian

            safe_divide_merged_logits(predicted_logits, n_predictions, bg_vec)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError("Encountered inf in predicted array. Aborting.")
        except Exception as e:
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits.cpu().float()

    def _predict_logits_roi_per_point_patch_single_fold(
        self,
        data: torch.Tensor,
        points_zyx: List[Tuple[int, int, int]],
        cfg: RoiPromptConfig,
    ) -> torch.Tensor:
        self.network = self.network.to(self.device)
        self.network.eval()
        shape = tuple(data.shape[1:])
        patch_size = tuple(self.configuration_manager.patch_size)
        prompt_cfg = cfg.prompt
        num_heads = self.label_manager.num_segmentation_heads

        data_f = data.float()
        data_padded, slicer_revert = pad_nd_image(
            data_f, patch_size, "constant", {"value": 0}, True, None
        )
        padded_shape = tuple(data_padded.shape[1:])
        data_padded = data_padded.to(self.device)

        results_device = self.device if self.perform_everything_on_device and self.device.type != "cpu" else torch.device("cpu")
        predicted_logits = torch.zeros(
            (num_heads, *padded_shape), dtype=torch.float32, device=results_device
        )
        n_predictions = torch.zeros(padded_shape, dtype=torch.float32, device=results_device)
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

        points_pad = map_points_zyx_unpadded_to_padded(points_zyx, slicer_revert)
        max_visits = cfg.inference.max_patch_expansion_visits
        max_depth = cfg.inference.max_patch_expansion_depth

        if not points_pad:
            cz, cy, cx = padded_shape[0] // 2, padded_shape[1] // 2, padded_shape[2] // 2
            sz, sy, sx = centered_spatial_slices_at_point(cz, cy, cx, patch_size, padded_shape)
            queue: deque = deque([(sz, sy, sx, None, 0)])
        else:
            seen_init = set()
            queue = deque()
            for seed in points_pad:
                sz, sy, sx = centered_spatial_slices_at_point(*seed, patch_size, padded_shape)
                k = (spatial_slices_to_tuple(sz, sy, sx), seed)
                if k in seen_init:
                    continue
                seen_init.add(k)
                queue.append((sz, sy, sx, seed, 0))

        visited = set()
        visit_count = 0
        bg_vec = background_logits_vector(self.label_manager, num_heads, results_device)

        try:
            while queue and visit_count < max_visits:
                sz, sy, sx, seed_pad, depth = queue.popleft()
                vkey = (spatial_slices_to_tuple(sz, sy, sx), seed_pad)
                if vkey in visited:
                    continue
                visited.add(vkey)
                visit_count += 1

                sl = (slice(None), sz, sy, sx)
                img_crop = data_padded[sl].clone().unsqueeze(0)
                local_pts = local_prompt_points_for_patch(seed_pad, sz, sy, sx, patch_size)
                prompt = encode_points_to_heatmap_pair(
                    local_pts,
                    [],
                    patch_size,
                    prompt_cfg.point_radius_vox,
                    prompt_cfg.encoding,
                    device=self.device,
                    intensity_scale=prompt_cfg.prompt_intensity_scale,
                )
                workon = torch.cat([img_crop[0], prompt], dim=0).unsqueeze(0).contiguous()
                pred_raw = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                pred = pred_raw * gaussian if self.use_gaussian else pred_raw
                predicted_logits[(slice(None), sz, sy, sx)] += pred
                n_predictions[sz, sy, sx] += gaussian

                if visit_count >= max_visits:
                    if queue:
                        self._roi_patch_visits_capped = True
                    break

                faces = touching_patch_faces_from_logits(pred_raw, self.label_manager)
                for axis, is_low in faces:
                    if max_depth is not None and depth >= max_depth:
                        continue
                    nsz, nsy, nsx = shift_spatial_slices(
                        sz, sy, sx, axis, is_low, patch_size, padded_shape
                    )
                    if spatial_slices_to_tuple(nsz, nsy, nsx) == spatial_slices_to_tuple(sz, sy, sx):
                        continue
                    nk = (spatial_slices_to_tuple(nsz, nsy, nsx), seed_pad)
                    if nk not in visited:
                        queue.append((nsz, nsy, nsx, seed_pad, depth + 1))

            safe_divide_merged_logits(predicted_logits, n_predictions, bg_vec)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError("Encountered inf in predicted array. Aborting.")
        except Exception as e:
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits.cpu().float()

    def _predict_logits_roi_single_fold(
        self,
        data: torch.Tensor,
        points_zyx: List[Tuple[int, int, int]],
        cfg: RoiPromptConfig,
        tile_step_size: float,
    ) -> torch.Tensor:
        self.network = self.network.to(self.device)
        self.network.eval()
        shape = tuple(data.shape[1:])
        patch_size = tuple(self.configuration_manager.patch_size)
        prompt_cfg = cfg.prompt
        prompt = encode_points_to_heatmap_pair(
            points_zyx,
            [],
            shape,
            prompt_cfg.point_radius_vox,
            prompt_cfg.encoding,
            device=self.device,
            intensity_scale=prompt_cfg.prompt_intensity_scale,
        )
        data = data.to(self.device)
        data_with_prompt = torch.cat([data, prompt], dim=0)
        data_padded, slicer_revert = pad_nd_image(
            data_with_prompt, patch_size, "constant", {"value": 0}, True, None
        )
        padded_shape = tuple(data_padded.shape[1:])
        slicers = get_prompt_aware_slicers(
            padded_shape, patch_size, tile_step_size, prompt[0:1].to(self.device)
        )
        if not slicers:
            return torch.zeros(
                (self.label_manager.num_segmentation_heads, *shape),
                dtype=torch.float32,
                device="cpu",
            )
        predicted_logits = self._internal_predict_sliding_window_return_logits(
            data_padded, slicers,
            self.perform_everything_on_device and self.device.type != "cpu",
        )
        predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError("Encountered inf in predicted array. Aborting.")
        return predicted_logits.cpu().float()

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Override to prevent accidental full-volume usage. ROI mode must use predict_logits_roi_mode.
        """
        raise RuntimeError(
            "nnUNetROIPredictor does not support full-volume sliding. Use predict_logits_roi_mode."
        )
