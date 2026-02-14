"""Large-lesion add-on: sparse extra patches for lesions larger than one patch."""
from typing import List, Tuple

import numpy as np

from nnunetv2.utilities.roi_config import LargeLesionConfig

try:
    import cc3d
except ImportError:
    cc3d = None


def get_lesion_bboxes_zyx(seg: np.ndarray) -> List[Tuple[int, int, int, int, int, int]]:
    """Per-instance bboxes as (zmin,zmax, ymin,ymax, xmin,xmax). cc3d returns (x,y,z)=(dim0,dim1,dim2)."""
    if cc3d is None:
        raise ImportError("connected-components-3d (cc3d) required")
    seg = np.asarray(seg)
    if seg.ndim == 4:
        seg = seg[0]
    if seg.ndim != 3:
        raise ValueError(f"seg must be 3D or 4D, got {seg.shape}")
    seg = seg.copy()
    seg[seg < 0] = 0
    if np.max(seg) == 0:
        return []
    seg_bin = (seg > 0).astype(np.uint8)
    labels = cc3d.connected_components(seg_bin)
    stats = cc3d.statistics(labels, no_slice_conversion=True)
    bboxes_raw = stats["bounding_boxes"]
    out: List[Tuple[int, int, int, int, int, int]] = []
    for i in range(1, bboxes_raw.shape[0]):
        d0min, d0max, d1min, d1max, d2min, d2max = bboxes_raw[i]
        out.append((int(d0min), int(d0max), int(d1min), int(d1max), int(d2min), int(d2max)))
    return out


def is_large_lesion(bbox_zyx: Tuple[int, int, int, int, int, int], patch_size: Tuple[int, ...]) -> bool:
    """True if any bbox extent exceeds patch size. bbox_zyx = (zmin,zmax, ymin,ymax, xmin,xmax)."""
    dz = bbox_zyx[1] - bbox_zyx[0]
    dy = bbox_zyx[3] - bbox_zyx[2]
    dx = bbox_zyx[5] - bbox_zyx[4]
    return any(d > p for d, p in zip((dz, dy, dx), patch_size[:3]))


def sample_extra_centers_for_large_lesion(
    seg: np.ndarray,
    bbox_zyx: Tuple[int, int, int, int, int, int],
    patch_size: Tuple[int, ...],
    cfg: LargeLesionConfig,
) -> List[Tuple[int, int, int]]:
    """Sample K centers from coarse grid inside lesion. Returns (z,y,x) in full-volume coords.
    Only keeps centers where the full patch fits inside the volume (no padding)."""
    seg = np.asarray(seg)
    if seg.ndim == 4:
        seg = seg[0]
    shape = seg.shape
    mask = seg > 0
    zmin, zmax, ymin, ymax, xmin, xmax = bbox_zyx
    strides = tuple(max(1, p // 2) for p in patch_size[:3])
    half = tuple(p // 2 for p in patch_size[:3])
    candidates: List[Tuple[int, int, int]] = []
    for z in range(zmin, zmax, strides[0]):
        for y in range(ymin, ymax, strides[1]):
            for x in range(xmin, xmax, strides[2]):
                if not mask[z, y, x]:
                    continue
                if z < half[0] or z >= shape[0] - half[0]:
                    continue
                if y < half[1] or y >= shape[1] - half[1]:
                    continue
                if x < half[2] or x >= shape[2] - half[2]:
                    continue
                candidates.append((z, y, x))
    if not candidates:
        return []
    k_lo, k_hi = cfg.K
    k_val = np.random.randint(k_lo, k_hi + 1) if k_lo != k_hi else k_lo
    k_eff = int(np.clip(k_val, cfg.K_min, cfg.K_max))
    n = min(k_eff, len(candidates))
    idx = np.random.choice(len(candidates), n, replace=False)
    return [candidates[i] for i in idx]
