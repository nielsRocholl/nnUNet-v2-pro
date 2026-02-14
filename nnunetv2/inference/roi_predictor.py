"""ROI-only inference: prompt-aware local sliding windows, no full-volume sliding."""
from typing import List, Optional, Tuple, Union

import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
from nnunetv2.utilities.prompt_encoding import encode_points_to_heatmap
from nnunetv2.utilities.roi_config import RoiPromptConfig


def get_prompt_aware_slicers(
    image_size: Tuple[int, ...],
    patch_size: Tuple[int, ...],
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
) -> Tuple[List[Tuple[int, int, int]], str]:
    """Load points from JSON. Returns (points_zyx, points_space)."""
    d = load_json(path)
    points = d.get("points")
    if points is None:
        raise KeyError("points_json must have 'points' key")
    points_space = points_space_override or d.get("points_space", "voxel")
    if points_space not in ("voxel", "world"):
        raise ValueError(f"points_space must be 'voxel' or 'world', got {points_space!r}")
    if points_space == "voxel":
        out = []
        for pt in points:
            if len(pt) != 3:
                raise ValueError(f"Point must have 3 coords, got {len(pt)}")
            z, y, x = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
            out.append((z, y, x))
        return out, points_space
    return [(float(p[0]), float(p[1]), float(p[2])) for p in points], points_space


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
        """Predict logits from preprocessed data using LesionLocator-style full-image sliding."""
        from torch._dynamo import OptimizedModule

        tile_step_size = tile_step_size if tile_step_size is not None else self.tile_step_size
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None
        for params in self.list_of_parameters:
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
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
        return prediction

    def _predict_logits_roi_single_fold(
        self,
        data: torch.Tensor,
        points_zyx: List[Tuple[int, int, int]],
        cfg: RoiPromptConfig,
        tile_step_size: float,
    ) -> torch.Tensor:
        shape = tuple(data.shape[1:])
        patch_size = tuple(self.configuration_manager.patch_size)
        prompt_cfg = cfg.prompt
        prompt = encode_points_to_heatmap(
            points_zyx,
            shape,
            prompt_cfg.point_radius_vox,
            prompt_cfg.encoding,
            device=self.device,
        ).unsqueeze(0)
        data_with_prompt = torch.cat([data, prompt], dim=0)
        data_padded, slicer_revert = pad_nd_image(
            data_with_prompt, patch_size, "constant", {"value": 0}, True, None
        )
        padded_shape = tuple(data_padded.shape[1:])
        slicers = get_prompt_aware_slicers(
            padded_shape, patch_size, tile_step_size, prompt.to(self.device)
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
