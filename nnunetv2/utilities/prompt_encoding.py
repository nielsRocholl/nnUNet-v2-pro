"""Prompt extraction and encoding: centroids from seg → heatmap channel [0,1]."""
from typing import List, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import ball

from nnunetv2.utilities.roi_config import PromptConfig, RoiPromptConfig

try:
    import cc3d
except ImportError:
    cc3d = None


def extract_centroids_from_seg(seg: np.ndarray) -> List[Tuple[int, int, int]]:
    """Extract lesion centroids from segmentation. Returns (z,y,x) per centroid.
    Handles binary and instance seg. LesionLocator logic via cc3d."""
    if cc3d is None:
        raise ImportError("connected-components-3d (cc3d) required for centroid extraction")
    seg = np.asarray(seg)
    if seg.ndim == 4:
        seg = seg[0]
    if seg.ndim != 3:
        raise ValueError(f"seg must be 3D or 4D, got shape {seg.shape}")
    seg = seg.copy()
    seg[seg < 0] = 0
    uniques = np.unique(seg)
    uniques = uniques[uniques > 0]
    if len(uniques) == 0:
        return []
    centroids: List[Tuple[int, int, int]] = []
    if len(uniques) == 1 and uniques[0] == 1:
        seg_bin = (seg > 0).astype(np.uint8)
        labels = cc3d.connected_components(seg_bin)
        stats = cc3d.statistics(labels, no_slice_conversion=True)
        for c in stats["centroids"][1:]:
            centroids.append((int(round(c[0])), int(round(c[1])), int(round(c[2]))))
    else:
        stats = cc3d.statistics(seg.astype(np.uint8), no_slice_conversion=True)
        for idx, c in enumerate(stats["centroids"][1:]):
            if idx + 1 in uniques:
                centroids.append((int(round(c[0])), int(round(c[1])), int(round(c[2]))))
    return centroids


def filter_centroids_in_patch(
    centroids: List[Tuple[int, int, int]],
    patch_slices: Tuple[slice, slice, slice],
) -> List[Tuple[int, int, int]]:
    """Keep centroids inside patch; convert to patch-local (z,y,x)."""
    slz, sly, slx = patch_slices
    out: List[Tuple[int, int, int]] = []
    for z, y, x in centroids:
        if slz.start <= z < slz.stop and sly.start <= y < sly.stop and slx.start <= x < slx.stop:
            out.append((z - slz.start, y - sly.start, x - slx.start))
    return out


def _build_ball_strel(radius: int, use_edt: bool) -> torch.Tensor:
    b = ball(radius, strict_radius=False)
    strel = torch.from_numpy(b.astype(np.float32))
    if use_edt:
        binary = (strel >= 0.5).numpy()
        edt = distance_transform_edt(binary)
        edt = edt / (edt.max() + 1e-8)
        strel = torch.from_numpy(edt.astype(np.float32))
    else:
        strel = (strel >= 0.5).float()
    return strel


def encode_points_to_heatmap(
    points_zyx: List[Tuple[int, int, int]],
    shape: Tuple[int, int, int],
    radius_vox: int,
    encoding: str,
    device: Union[torch.device, str, None] = None,
) -> torch.Tensor:
    """Build heatmap [0,1] from points. Merge multiple with torch.maximum."""
    heatmap = torch.zeros(shape, dtype=torch.float32, device=device)
    if not points_zyx:
        return heatmap
    use_edt = encoding == "edt"
    strel = _build_ball_strel(radius_vox, use_edt)
    r = radius_vox
    for pz, py, px in points_zyx:
        z0 = max(0, pz - r)
        z1 = min(shape[0], pz + r + 1)
        y0 = max(0, py - r)
        y1 = min(shape[1], py + r + 1)
        x0 = max(0, px - r)
        x1 = min(shape[2], px + r + 1)
        sz0 = r - (pz - z0)
        sz1 = sz0 + (z1 - z0)
        sy0 = r - (py - y0)
        sy1 = sy0 + (y1 - y0)
        sx0 = r - (px - x0)
        sx1 = sx0 + (x1 - x0)
        slc = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        strel_slc = (slice(sz0, sz1), slice(sy0, sy1), slice(sx0, sx1))
        patch = strel[strel_slc].to(heatmap.device)
        torch.maximum(heatmap[slc], patch, out=heatmap[slc])
    return heatmap


def build_prompt_channel(
    seg: np.ndarray,
    patch_slices: Tuple[slice, slice, slice],
    cfg: Union[PromptConfig, RoiPromptConfig],
    device: Union[torch.device, str, None] = None,
) -> torch.Tensor:
    """Full pipeline: extract centroids → filter to patch → encode. Returns (1, D, H, W) float32."""
    prompt_cfg = cfg.prompt if hasattr(cfg, "prompt") else cfg
    patch_shape = (
        patch_slices[0].stop - patch_slices[0].start,
        patch_slices[1].stop - patch_slices[1].start,
        patch_slices[2].stop - patch_slices[2].start,
    )
    centroids = extract_centroids_from_seg(seg)
    local = filter_centroids_in_patch(centroids, patch_slices)
    heatmap = encode_points_to_heatmap(
        local,
        patch_shape,
        prompt_cfg.point_radius_vox,
        prompt_cfg.encoding,
        device,
    )
    return heatmap.unsqueeze(0)
