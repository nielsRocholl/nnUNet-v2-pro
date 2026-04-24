"""Greedy cluster of 3D points: same cluster iff axis-aligned bbox (+ margin) fits in patch_size."""
from typing import List, Tuple

ZYX = Tuple[int, int, int]


def _margin_vox(patch_size: Tuple[int, int, int], margin_frac: float) -> Tuple[int, int, int]:
    m = max(0.0, min(0.5, float(margin_frac)))
    return (
        int(round(m * patch_size[0])),
        int(round(m * patch_size[1])),
        int(round(m * patch_size[2])),
    )


def bbox_fits_in_patch(
    points: List[ZYX],
    patch_size: Tuple[int, int, int],
    margin: Tuple[int, int, int],
) -> bool:
    if not points:
        return True
    z = [p[0] for p in points]
    y = [p[1] for p in points]
    x = [p[2] for p in points]
    e0 = max(z) - min(z) + 1 + 2 * margin[0]
    e1 = max(y) - min(y) + 1 + 2 * margin[1]
    e2 = max(x) - min(x) + 1 + 2 * margin[2]
    return e0 <= patch_size[0] and e1 <= patch_size[1] and e2 <= patch_size[2]


def cluster_points_for_patch_size(
    points: List[ZYX],
    patch_size: Tuple[int, int, int],
    margin_frac: float = 0.1,
) -> List[List[ZYX]]:
    """Sort by z, then y, x; assign each point to first cluster whose bbox+margin still fits, else new cluster."""
    if not points:
        return []
    margin = _margin_vox(patch_size, margin_frac)
    sorted_pts = sorted(points, key=lambda t: (t[0], t[1], t[2]))
    clusters: List[List[ZYX]] = []
    for p in sorted_pts:
        for c in clusters:
            if bbox_fits_in_patch(c + [p], patch_size, margin):
                c.append(p)
                break
        else:
            clusters.append([p])
    return clusters


def cluster_centroid_zyx(points: List[ZYX]) -> ZYX:
    n = len(points)
    if n == 0:
        raise ValueError("empty cluster")
    return (
        int(round(sum(p[0] for p in points) / n)),
        int(round(sum(p[1] for p in points) / n)),
        int(round(sum(p[2] for p in points) / n)),
    )
