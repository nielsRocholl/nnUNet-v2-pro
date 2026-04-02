"""Tests for ROI coordinate validation and conversion."""
import json
import os
import tempfile

import pytest

from nnunetv2.utilities.roi_coordinate_validation import validate_and_convert_points


def test_zyx_voxel_no_op():
    pts = validate_and_convert_points([[10, 20, 30], [0, 0, 0]], "voxel", "zyx_voxel")
    assert pts == [(10, 20, 30), (0, 0, 0)]


def test_xyz_voxel_reorder():
    pts = validate_and_convert_points([[10, 20, 30]], "voxel", "xyz_voxel")
    assert pts == [(30, 20, 10)]


def test_xyz_world_no_op():
    pts = validate_and_convert_points([[16.0, 32.0, 31.0]], "world", "xyz_world")
    assert pts == [(16.0, 32.0, 31.0)]


def test_zyx_world_reorder():
    pts = validate_and_convert_points([[31.0, 32.0, 16.0]], "world", "zyx_world")
    assert pts == [(16.0, 32.0, 31.0)]


def test_default_format_voxel():
    pts = validate_and_convert_points([[60, 125, 125]], "voxel", None)
    assert pts == [(60, 125, 125)]


def test_default_format_world():
    pts = validate_and_convert_points([[100.0, 200.0, 300.0]], "world", None)
    assert pts == [(100.0, 200.0, 300.0)]


def test_invalid_point_length():
    with pytest.raises(ValueError, match="3 coords"):
        validate_and_convert_points([[1, 2]], "voxel", "zyx_voxel")


def test_invalid_non_numeric():
    with pytest.raises(ValueError, match="numeric"):
        validate_and_convert_points([["a", 2, 3]], "voxel", "zyx_voxel")


def test_invalid_format():
    with pytest.raises(ValueError, match="points_format"):
        validate_and_convert_points([[1, 2, 3]], "voxel", "invalid")


def test_format_space_mismatch():
    with pytest.raises(ValueError, match="requires points_space='voxel'"):
        validate_and_convert_points([[1, 2, 3]], "world", "zyx_voxel")
    with pytest.raises(ValueError, match="requires points_space='world'"):
        validate_and_convert_points([[1, 2, 3]], "voxel", "xyz_world")


def test_round_trip_xyz_voxel():
    zyx = (30, 20, 10)
    xyz = [10, 20, 30]
    converted = validate_and_convert_points([xyz], "voxel", "xyz_voxel")
    assert converted[0] == zyx


def test_round_trip_zyx_world():
    xyz = (16.0, 32.0, 31.0)
    zyx = [31.0, 32.0, 16.0]
    converted = validate_and_convert_points([zyx], "world", "zyx_world")
    assert converted[0] == xyz


@pytest.mark.slow
def test_roi_inference_all_formats():
    """Integration: ROI inference with each format on Dataset010 or Dataset010_CECT."""
    import sys

    from batchgenerators.utilities.file_and_folder_operations import join, load_json

    from nnunetv2.inference.roi_predict_entrypoint import predict_roi_entry_point
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder

    raw = os.environ.get("nnUNet_raw", "")
    preprocessed = os.environ.get("nnUNet_preprocessed", "")
    for ds_name in ("Dataset010_CECT", "Dataset010"):
        images_dir = join(raw, ds_name, "imagesTr")
        preproc_dir = join(preprocessed, ds_name, "nnUNetPlans_3d_fullres")
        if not os.path.isdir(images_dir) or not os.path.isdir(preproc_dir):
            continue
        model_folder = get_output_folder(ds_name, "nnUNetTrainerPromptAware", "nnUNetPlans", "3d_fullres")
        if not os.path.isdir(model_folder):
            continue
        try:
            import blosc2
        except ImportError:
            pytest.skip("blosc2 required")
        ids = [i[:-5] for i in os.listdir(preproc_dir) if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")]
        if not ids:
            continue
        ident = ids[0]
        data = __import__("numpy").array(blosc2.open(join(preproc_dir, ident + ".b2nd"), mode="r"))
        shape = data.shape[1:]
        center_zyx = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
        center_xyz = [center_zyx[2], center_zyx[1], center_zyx[0]]
        cfg = join(os.path.dirname(__file__), "fixtures", "nnunet_pro_config.json")
        if not os.path.isfile(cfg):
            pytest.skip("fixture config not found")
        dataset_json = load_json(join(model_folder, "dataset.json"))
        caseids = get_identifiers_from_splitted_dataset_folder(images_dir, dataset_json["file_ending"])
        if not caseids:
            continue
        expected_caseid = caseids[0]

        for fmt, points in [
            ("zyx_voxel", [list(center_zyx)]),
            ("xyz_voxel", [center_xyz]),
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                points_path = join(tmp, "points.json")
                with open(points_path, "w") as f:
                    json.dump({"points": points, "points_space": "voxel", "points_format": fmt}, f)
                out_dir = join(tmp, "out")
                os.makedirs(out_dir, exist_ok=True)
                old_argv = sys.argv
                sys.argv = [
                    "nnUNetv2_predict_roi",
                    "-i", images_dir,
                    "-o", out_dir,
                    "-m", model_folder,
                    "--config", cfg,
                    "--points_json", points_path,
                    "-f", "0", "-device", "cpu", "-npp", "1", "-nps", "1",
                ]
                try:
                    predict_roi_entry_point()
                finally:
                    sys.argv = old_argv
                file_ending = dataset_json["file_ending"]
                out_file = join(out_dir, expected_caseid + file_ending)
                assert os.path.isfile(out_file), f"Format {fmt}: expected output {out_file}"
        return
    pytest.skip("Dataset010 or Dataset010_CECT with nnUNetTrainerPromptAware not found")
