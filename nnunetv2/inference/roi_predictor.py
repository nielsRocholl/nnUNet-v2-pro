"""ROI-only inference: prompt-aware local sliding windows, no full-volume sliding."""
from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.helpers import empty_cache
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


def parse_points_json(
    path: str,
    points_space_override: Optional[str] = None,
) -> Tuple[List, str, Optional[str]]:
    """Load points from JSON. Returns (points_raw, points_space, points_format)."""
    d = load_json(path)
    points = d.get("points")
    if points is None:
        raise KeyError("points_json must have 'points' key")
    points_space = points_space_override or d.get("points_space", "voxel")
    if points_space not in ("voxel", "world"):
        raise ValueError(f"points_space must be 'voxel' or 'world', got {points_space!r}")
    points_format = d.get("points_format")
    if points_format is None:
        points_format = "zyx_voxel" if points_space == "voxel" else "xyz_world"
    return points, points_space, points_format


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
