"""Coordinate conversion: points (voxel/world) → (z,y,x) in preprocessed space."""
from typing import List, Tuple, Union

import numpy as np


def points_to_centers_zyx(
    points: List,
    points_space: str,
    properties: dict,
    preprocessed_shape: Tuple[int, ...],
    spacing: Tuple[float, ...],
    transpose_forward: Union[Tuple[int, ...], List[int], None] = None,
) -> List[Tuple[int, int, int]]:
    """Convert points to (z,y,x) integer centers in preprocessed voxel space."""
    if points_space not in ("voxel", "world"):
        raise ValueError(f"points_space must be 'voxel' or 'world', got {points_space!r}")
    out = []
    for pt in points:
        if len(pt) != 3:
            raise ValueError(f"Point must have 3 coords, got {len(pt)}")
        if points_space == "voxel":
            z, y, x = np.round(pt).astype(int)
        else:
            x_phys, y_phys, z_phys = float(pt[0]), float(pt[1]), float(pt[2])
            sitk = properties.get("sitk_stuff")
            if not sitk:
                raise KeyError("points_space='world' requires properties['sitk_stuff'] (origin, spacing)")
            orig = sitk["origin"]
            sp = sitk["spacing"]
            vox_x = (x_phys - orig[0]) / sp[0]
            vox_y = (y_phys - orig[1]) / sp[1]
            vox_z = (z_phys - orig[2]) / sp[2]
            z, y, x = vox_z, vox_y, vox_x
            if transpose_forward is not None:
                arr = np.array([z, y, x])
                z, y, x = arr[transpose_forward[0]], arr[transpose_forward[1]], arr[transpose_forward[2]]
            bbox = properties["bbox_used_for_cropping"]
            shape_ac = properties["shape_after_cropping_and_before_resampling"]
            x = max(0, min(shape_ac[2] - 1, x - bbox[2][0]))
            y = max(0, min(shape_ac[1] - 1, y - bbox[1][0]))
            z = max(0, min(shape_ac[0] - 1, z - bbox[0][0]))
            factor = [preprocessed_shape[i] / shape_ac[i] for i in range(3)]
            x = np.round(x * factor[2]).astype(int)
            y = np.round(y * factor[1]).astype(int)
            z = np.round(z * factor[0]).astype(int)
        z = int(np.clip(z, 0, preprocessed_shape[0] - 1))
        y = int(np.clip(y, 0, preprocessed_shape[1] - 1))
        x = int(np.clip(x, 0, preprocessed_shape[2] - 1))
        out.append((z, y, x))
    return out
