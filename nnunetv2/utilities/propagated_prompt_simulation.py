"""Simulate propagated COG prompts: centroid + random offset from propagation error distribution."""
from typing import Tuple

import numpy as np


def apply_propagation_offset(
    centroid_zyx: Tuple[int, int, int],
    patch_shape: Tuple[int, int, int],
    sigma_per_axis: Tuple[float, float, float],
    max_vox: float,
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    """Perturb centroid by anisotropic Gaussian offset, truncated and clipped to patch.

    Offsets follow N(0, diag(σ_z², σ_y², σ_x²)). Magnitude truncated to max_vox.
    Result clipped to patch bounds [0, D-1] x [0, H-1] x [0, W-1].
    """
    cz, cy, cx = centroid_zyx
    dz, dy, dx = rng.normal(0, sigma_per_axis)
    mag = np.sqrt(dz * dz + dy * dy + dx * dx)
    if mag > max_vox and mag > 1e-8:
        scale = max_vox / mag
        dz, dy, dx = dz * scale, dy * scale, dx * scale
    pz = int(round(cz + dz))
    py = int(round(cy + dy))
    px = int(round(cx + dx))
    d, h, w = patch_shape
    pz = max(0, min(d - 1, pz))
    py = max(0, min(h - 1, py))
    px = max(0, min(w - 1, px))
    return (pz, py, px)
