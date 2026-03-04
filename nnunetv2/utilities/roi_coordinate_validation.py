"""Validate and convert ROI prompt coordinates to canonical form (z,y,x) voxel or (x,y,z) world."""
from typing import List, Optional, Tuple, Union


VALID_FORMATS = frozenset({"zyx_voxel", "xyz_voxel", "xyz_world", "zyx_world"})


def validate_and_convert_points(
    points_raw: List,
    points_space: str,
    points_format: Optional[str] = None,
) -> List[Union[Tuple[int, int, int], Tuple[float, float, float]]]:
    """Validate and convert to canonical form. Raises on invalid input.
    Returns (z,y,x) for voxel or (x,y,z) for world."""
    if points_space not in ("voxel", "world"):
        raise ValueError(f"points_space must be 'voxel' or 'world', got {points_space!r}")
    fmt = points_format or ("zyx_voxel" if points_space == "voxel" else "xyz_world")
    if fmt not in VALID_FORMATS:
        raise ValueError(f"points_format must be one of {sorted(VALID_FORMATS)}, got {fmt!r}")
    if fmt.endswith("_voxel") and points_space != "voxel":
        raise ValueError(f"points_format {fmt!r} requires points_space='voxel', got {points_space!r}")
    if fmt.endswith("_world") and points_space != "world":
        raise ValueError(f"points_format {fmt!r} requires points_space='world', got {points_space!r}")

    out = []
    for pt in points_raw:
        if len(pt) != 3:
            raise ValueError(f"Point must have 3 coords, got {len(pt)}")
        try:
            p = [float(pt[0]), float(pt[1]), float(pt[2])]
        except (TypeError, ValueError) as e:
            raise ValueError(f"Point coords must be numeric, got {pt}") from e

        if fmt == "zyx_voxel":
            out.append((int(round(p[0])), int(round(p[1])), int(round(p[2]))))
        elif fmt == "xyz_voxel":
            out.append((int(round(p[2])), int(round(p[1])), int(round(p[0]))))
        elif fmt == "xyz_world":
            out.append((p[0], p[1], p[2]))
        else:
            assert fmt == "zyx_world"
            out.append((p[2], p[1], p[0]))
    return out
