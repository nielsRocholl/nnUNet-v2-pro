import pytest

from nnunetv2.inference.prompt_clustering import (
    bbox_fits_in_patch,
    cluster_centroid_zyx,
    cluster_points_for_patch_size,
)


def test_bbox_fits_empty():
    assert bbox_fits_in_patch([], (8, 8, 8), (0, 0, 0)) is True


def test_bbox_fits_one():
    assert bbox_fits_in_patch([(3, 3, 3)], (8, 8, 8), (0, 0, 0)) is True


def test_bbox_too_long_axis():
    ps = (8, 8, 8)
    m = (0, 0, 0)
    # span z: 0..7 = 8 voxels, fits exactly
    assert bbox_fits_in_patch([(0, 0, 0), (7, 0, 0)], ps, m) is True
    # span 9 voxels
    assert bbox_fits_in_patch([(0, 0, 0), (8, 0, 0)], ps, m) is False


def test_cluster_single_point():
    c = cluster_points_for_patch_size([(1, 2, 3)], (16, 16, 16), 0.1)
    assert len(c) == 1 and c[0] == [(1, 2, 3)]


def test_cluster_two_close_one_cluster():
    ps = (16, 16, 16)
    c = cluster_points_for_patch_size([(0, 0, 0), (1, 0, 0)], ps, 0.0)
    assert len(c) == 1
    assert len(c[0]) == 2


def test_cluster_two_far_apart():
    ps = (8, 8, 8)
    m = 0.0
    a = (0, 0, 0)
    b = (20, 0, 0)
    assert bbox_fits_in_patch([a, b], ps, (0, 0, 0)) is False
    c = cluster_points_for_patch_size([a, b], ps, m)
    assert len(c) == 2


def test_cluster_centroid():
    assert cluster_centroid_zyx([(0, 0, 0), (2, 0, 0)]) == (1, 0, 0)


def test_cluster_centroid_empty_raises():
    with pytest.raises(ValueError):
        cluster_centroid_zyx([])
