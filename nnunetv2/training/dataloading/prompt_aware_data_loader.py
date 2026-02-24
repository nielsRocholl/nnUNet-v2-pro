"""Prompt-aware training dataloader: stochastic patch sampling with four modes."""
from typing import List, Tuple, Union

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.large_lesion_sampling import (
    get_lesion_bboxes_zyx,
    is_large_lesion,
    sample_extra_centers_for_large_lesion,
)
from nnunetv2.utilities.propagated_prompt_simulation import apply_propagation_offset
from nnunetv2.utilities.prompt_encoding import (
    encode_points_to_heatmap,
    extract_centroids_from_seg,
    filter_centroids_in_patch,
)
from nnunetv2.utilities.roi_config import RoiPromptConfig

MODE_POS, MODE_POS_SPUR, MODE_POS_NO_PROMPT, MODE_NEG = 0, 1, 2, 3


def _sample_spurious(seg: np.ndarray, n: int) -> List[Tuple[int, int, int]]:
    """Sample n points from B (non-lesion voxels). seg: (C,D,H,W) or (D,H,W)."""
    s = np.asarray(seg)
    if s.ndim == 4:
        s = s[0]
    bg = (s == 0)
    coords = np.argwhere(bg)
    if len(coords) < n:
        n = len(coords)
    if n == 0:
        return []
    idx = np.random.choice(len(coords), n, replace=False)
    return [tuple(int(c) for c in coords[i]) for i in idx]


def _sample_negative(shape: Tuple[int, int, int], n: int) -> List[Tuple[int, int, int]]:
    """Sample n points uniformly from Ω (all patch voxels)."""
    total = shape[0] * shape[1] * shape[2]
    if total == 0 or n == 0:
        return []
    n = min(n, total)
    flat = np.random.choice(total, n, replace=False)
    z = flat // (shape[1] * shape[2])
    r = flat % (shape[1] * shape[2])
    y = r // shape[2]
    x = r % shape[2]
    return [(int(z[i]), int(y[i]), int(x[i])) for i in range(n)]


def _lesion_voxels(seg: np.ndarray) -> np.ndarray:
    """Return (N,3) array of lesion voxel coords (z,y,x)."""
    s = np.asarray(seg)
    if s.ndim == 4:
        s = s[0]
    fg = (s > 0)
    return np.argwhere(fg)


def _has_lesion(seg: np.ndarray) -> bool:
    s = np.asarray(seg)
    if s.ndim == 4:
        s = s[0]
    return np.any(s > 0)


class nnUNetPromptAwareDataLoader(nnUNetDataLoader):
    """Stochastic patch sampler with prompt-aware modes (pos, pos+spurious, pos+no-prompt, negative)."""

    def __init__(
        self,
        data: nnUNetBaseDataset,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        label_manager: LabelManager,
        cfg: RoiPromptConfig,
        oversample_foreground_percent: float = 0.0,
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
        pad_sides: Union[List[int], Tuple[int, ...]] = None,
        probabilistic_oversampling: bool = False,
        transforms=None,
        force_zero_prompt: bool = False,
    ):
        super().__init__(
            data, batch_size, patch_size, final_patch_size, label_manager,
            oversample_foreground_percent, sampling_probabilities, pad_sides,
            probabilistic_oversampling, transforms,
        )
        self.cfg = cfg
        self.force_zero_prompt = force_zero_prompt

    def determine_shapes(self):
        data_shape, seg_shape = super().determine_shapes()
        c = data_shape[1]
        new_data_shape = (data_shape[0], c + 1, *data_shape[2:])
        return new_data_shape, seg_shape

    def _get_bbox_and_mode(
        self,
        shape: np.ndarray,
        class_locations: Union[dict, None],
    ) -> Tuple[List[int], List[int], int]:
        probs = np.array(self.cfg.sampling.mode_probs)
        mode = int(np.random.choice(4, p=probs))
        eligible = []
        if class_locations:
            eligible = [
                k for k in class_locations
                if k != self.annotated_classes_key and len(class_locations[k]) > 0
            ]
        has_fg = len(eligible) > 0
        if mode == MODE_NEG and not has_fg:
            mode = MODE_POS
        elif mode != MODE_NEG and not has_fg:
            mode = MODE_NEG
        force_fg = mode != MODE_NEG
        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locations)
        return bbox_lbs, bbox_ubs, mode

    def _sample_large_lesion_extras(
        self, selected_keys: List
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        ll_cfg = self.cfg.sampling.large_lesion
        if ll_cfg.max_extra == 0:
            empty_d = np.zeros((0,) + self.data_shape[1:], dtype=np.float32)
            empty_s = np.zeros((0,) + self.seg_shape[1:], dtype=np.int16)
            return empty_d, empty_s, []
        patch_size = tuple(self.patch_size)
        extra_data_list, extra_seg_list, extra_keys_list = [], [], []
        for case_id in selected_keys:
            data, seg, seg_prev, _ = self._data.load_case(case_id)
            bboxes = get_lesion_bboxes_zyx(seg)
            all_centers: List[Tuple[int, int, int]] = []
            for bbox in bboxes:
                if not is_large_lesion(bbox, patch_size):
                    continue
                centers = sample_extra_centers_for_large_lesion(
                    seg, bbox, patch_size, ll_cfg
                )
                all_centers.extend(centers)
            np.random.shuffle(all_centers)
            for center in all_centers[: ll_cfg.max_extra]:
                cz, cy, cx = center
                bbox_lbs = [
                    cz - patch_size[0] // 2,
                    cy - patch_size[1] // 2,
                    cx - patch_size[2] // 2,
                ]
                bbox_ubs = [
                    bbox_lbs[0] + patch_size[0],
                    bbox_lbs[1] + patch_size[1],
                    bbox_lbs[2] + patch_size[2],
                ]
                bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
                data_crop = np.asarray(crop_and_pad_nd(data, bbox, 0))
                seg_crop = np.asarray(crop_and_pad_nd(seg, bbox, -1))
                if seg_prev is not None:
                    seg_prev_crop = np.asarray(crop_and_pad_nd(seg_prev, bbox, -1))
                    seg_crop = np.vstack((seg_crop, seg_prev_crop[None]))
                patch_shape = tuple(patch_size)
                slz = slice(bbox_lbs[0], bbox_ubs[0])
                sly = slice(bbox_lbs[1], bbox_ubs[1])
                slx = slice(bbox_lbs[2], bbox_ubs[2])
                patch_slices = (slz, sly, slx)
                centroids = extract_centroids_from_seg(seg_crop)
                points = filter_centroids_in_patch(centroids, patch_slices)
                if not points:
                    les = _lesion_voxels(seg_crop)
                    if len(les) > 0:
                        idx = np.random.randint(len(les))
                        points = [tuple(int(x) for x in les[idx])]
                prop = self.cfg.sampling.propagated
                rng = np.random.default_rng()
                points = [
                    apply_propagation_offset(
                        p, patch_shape, prop.sigma_per_axis, prop.max_vox, rng,
                    )
                    for p in points
                ]
                heatmap = encode_points_to_heatmap(
                    points, patch_shape,
                    self.cfg.prompt.point_radius_vox, self.cfg.prompt.encoding,
                    device=None,
                )
                data_with_prompt = np.concatenate([data_crop, heatmap.numpy()[None]], axis=0)
                extra_data_list.append(data_with_prompt)
                extra_seg_list.append(seg_crop)
                extra_keys_list.append(case_id)
        if not extra_data_list:
            empty_d = np.zeros((0,) + self.data_shape[1:], dtype=np.float32)
            empty_s = np.zeros((0,) + self.seg_shape[1:], dtype=np.int16)
            return empty_d, empty_s, []
        return (
            np.stack(extra_data_list).astype(np.float32),
            np.stack(extra_seg_list).astype(np.int16),
            extra_keys_list,
        )

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        prompt_cfg = self.cfg.prompt

        for j, i in enumerate(selected_keys):
            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            bbox_lbs, bbox_ubs, mode = self._get_bbox_and_mode(shape, properties.get("class_locations"))
            bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

            data_crop = np.asarray(crop_and_pad_nd(data, bbox, 0))
            seg_crop = np.asarray(crop_and_pad_nd(seg, bbox, -1))
            if seg_prev is not None:
                seg_prev_crop = np.asarray(crop_and_pad_nd(seg_prev, bbox, -1))
                seg_crop = np.vstack((seg_crop, seg_prev_crop[None]))

            patch_shape = tuple(bbox_ubs[k] - bbox_lbs[k] for k in range(len(bbox_lbs)))
            slz = slice(bbox_lbs[0], bbox_ubs[0])
            sly = slice(bbox_lbs[1], bbox_ubs[1])
            slx = slice(bbox_lbs[2], bbox_ubs[2])
            patch_slices = (slz, sly, slx)

            if mode == MODE_NEG and _has_lesion(seg_crop):
                mode = MODE_POS
            elif mode != MODE_NEG and not _has_lesion(seg_crop):
                mode = MODE_NEG

            if self.force_zero_prompt:
                points: List[Tuple[int, int, int]] = []
            elif mode == MODE_POS_NO_PROMPT:
                points = []
            elif mode == MODE_NEG:
                n_neg = np.random.randint(self.cfg.sampling.n_neg[0], self.cfg.sampling.n_neg[1] + 1)
                points = _sample_negative(patch_shape, n_neg)
            else:
                centroids = extract_centroids_from_seg(seg_crop)
                points = filter_centroids_in_patch(centroids, patch_slices)
                if not points:
                    les = _lesion_voxels(seg_crop)
                    if len(les) > 0:
                        idx = np.random.randint(len(les))
                        points = [tuple(int(x) for x in les[idx])]
                prop = self.cfg.sampling.propagated
                rng = np.random.default_rng()
                points = [
                    apply_propagation_offset(
                        p, patch_shape, prop.sigma_per_axis, prop.max_vox, rng,
                    )
                    for p in points
                ]
                if mode == MODE_POS_SPUR:
                    n_spur = np.random.randint(self.cfg.sampling.n_spur[0], self.cfg.sampling.n_spur[1] + 1)
                    spur = _sample_spurious(seg_crop, n_spur)
                    points = points + spur

            heatmap = encode_points_to_heatmap(
                points, patch_shape,
                prompt_cfg.point_radius_vox, prompt_cfg.encoding,
                device=None,
            )
            prompt_np = heatmap.numpy()
            data_with_prompt = np.concatenate([data_crop, prompt_np[None]], axis=0)
            data_all[j] = data_with_prompt
            seg_all[j] = seg_crop

        if not self.force_zero_prompt:
            extra_data, extra_seg, extra_keys = self._sample_large_lesion_extras(selected_keys)
            if len(extra_keys) > 0:
                data_all = np.concatenate([data_all, extra_data], axis=0)
                seg_all = np.concatenate([seg_all, extra_seg], axis=0)
                selected_keys = list(selected_keys) + extra_keys

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images, segs = [], []
                    for b in range(data_all.shape[0]):
                        tmp = self.transforms(**{"image": data_all[b], "segmentation": seg_all[b]})
                        images.append(tmp["image"])
                        segs.append(tmp["segmentation"])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
            return {"data": data_all, "target": seg_all, "keys": selected_keys}
        return {"data": data_all, "target": seg_all, "keys": selected_keys}
