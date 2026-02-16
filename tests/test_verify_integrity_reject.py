"""Test relaxed geometry tolerance and reject_failing_cases in verify_dataset_integrity."""
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_json
from nnunetv2.experiment_planning.verify_dataset_integrity import (
    _GEOM_ATOL,
    _GEOM_RTOL,
    check_cases,
    verify_dataset_integrity,
)
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

ULS_NNUNET_RAW = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/universal-lesion-segmentation/nnUNet_raw"


def _mock_reader(img_spacing, seg_spacing, img_shape=(1, 32, 32, 32), seg_shape=(1, 32, 32, 32)):
    """Mock reader class returning controlled spacing for image and seg. check_cases instantiates readerclass()."""
    instance = MagicMock()
    img_data = np.zeros(img_shape, dtype=np.float32)
    seg_data = np.zeros(seg_shape, dtype=np.uint8)
    img_props = {"spacing": np.array(img_spacing), "sitk_stuff": {"origin": (0, 0, 0), "direction": np.eye(3).ravel()}}
    seg_props = {"spacing": np.array(seg_spacing), "sitk_stuff": {"origin": (0, 0, 0), "direction": np.eye(3).ravel()}}
    instance.read_images.return_value = (img_data, img_props)
    instance.read_seg.return_value = (seg_data, seg_props)
    return MagicMock(return_value=instance)


def test_relaxed_tolerance_minuscule_spacing_passes():
    """Minuscule spacing diff (0.8 vs 0.80004883) should pass with _GEOM_RTOL/_GEOM_ATOL."""
    reader = _mock_reader([0.8, 0.7, 0.7], [0.80004883, 0.7, 0.7])
    ok, errors = check_cases(["img.nii.gz"], "seg.nii.gz", 1, reader)
    assert ok, f"Expected pass, got errors: {errors}"


def test_relaxed_tolerance_origin_passes():
    """Small origin diff should pass."""
    reader = _mock_reader([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    reader.return_value.read_images.return_value = (
        np.zeros((1, 10, 10, 10), dtype=np.float32),
        {"spacing": [1, 1, 1], "sitk_stuff": {"origin": (0.0, 0.0, 0.0), "direction": np.eye(3).ravel()}},
    )
    reader.return_value.read_seg.return_value = (
        np.zeros((1, 10, 10, 10), dtype=np.uint8),
        {"spacing": [1, 1, 1], "sitk_stuff": {"origin": (0.0, 0.00005, 0.0), "direction": np.eye(3).ravel()}},
    )
    ok, errors = check_cases(["img.nii.gz"], "seg.nii.gz", 1, reader)
    assert ok, f"Expected pass, got errors: {errors}"


def test_strict_spacing_mismatch_fails():
    """Larger spacing diff (0.8 vs 0.9) should fail."""
    reader = _mock_reader([0.8, 0.7, 0.7], [0.9, 0.7, 0.7])
    ok, errors = check_cases(["img.nii.gz"], "seg.nii.gz", 1, reader)
    assert not ok
    assert any("Spacing mismatch" in e for e in errors)


def test_shape_mismatch_fails():
    """Shape mismatch should always fail."""
    reader = _mock_reader([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], img_shape=(1, 32, 32, 32), seg_shape=(1, 30, 30, 30))
    ok, errors = check_cases(["img.nii.gz"], "seg.nii.gz", 1, reader)
    assert not ok
    assert any("Shape mismatch" in e for e in errors)


def test_geom_tolerance_constants():
    """Verify tolerance constants allow typical float noise."""
    a, b = np.array([0.8, 0.7, 0.7]), np.array([0.80004883, 0.69999999, 0.7])
    assert np.allclose(a, b, rtol=_GEOM_RTOL, atol=_GEOM_ATOL)


@pytest.mark.slow
def test_reject_failing_cases_moves_to_rejected(tmp_path):
    """With reject_failing_cases, failing cases move to imagesTr_rejected/labelsTr_rejected."""
    if not os.path.isdir(ULS_NNUNET_RAW):
        pytest.skip(f"ULS data not found at {ULS_NNUNET_RAW}")

    try:
        import SimpleITK as sitk
    except ImportError:
        pytest.skip("SimpleITK required for integration test")

    raw = tmp_path / "nnUNet_raw"
    raw.mkdir()
    ds_src = Path(ULS_NNUNET_RAW) / "Dataset010_CECT"
    if not ds_src.is_dir():
        pytest.skip("Dataset010_CECT not found")
    ds_dst = raw / "Dataset001_RejectTest"
    maybe_mkdir_p(str(ds_dst))
    maybe_mkdir_p(str(ds_dst / "imagesTr"))
    maybe_mkdir_p(str(ds_dst / "labelsTr"))

    ds_json_src = load_json(ds_src / "dataset.json")
    ds = get_filenames_of_train_images_and_targets(str(ds_src), ds_json_src)
    keys = list(ds.keys())[:2]
    if len(keys) < 2:
        pytest.skip("Need at least 2 cases in Dataset010_CECT")

    subset = {}
    for i, k in enumerate(keys):
        img_files = ds[k]["images"]
        lbl_file = ds[k]["label"]
        for j, img in enumerate(img_files):
            dst_img = ds_dst / "imagesTr" / f"{k}_000{j}.nii.gz"
            if Path(img).exists():
                sitk.WriteImage(sitk.ReadImage(img), str(dst_img))
        dst_lbl = ds_dst / "labelsTr" / f"{k}.nii.gz"
        if Path(lbl_file).exists():
            lbl_img = sitk.ReadImage(lbl_file)
            if i == 1:
                lbl_img.SetSpacing((1.5, 1.5, 1.5))
            sitk.WriteImage(lbl_img, str(dst_lbl))
        rel_imgs = [f"imagesTr/{k}_000{j}.nii.gz" for j in range(len(img_files))]
        subset[k] = {"images": rel_imgs, "label": f"labelsTr/{k}.nii.gz"}

    out_json = {
        "channel_names": ds_json_src["channel_names"],
        "labels": ds_json_src["labels"],
        "file_ending": ds_json_src["file_ending"],
        "numTraining": 2,
        "dataset": subset,
    }
    save_json(out_json, ds_dst / "dataset.json", sort_keys=False)

    old_raw = os.environ.get("nnUNet_raw")
    os.environ["nnUNet_raw"] = str(raw)
    try:
        verify_dataset_integrity(str(ds_dst), num_processes=1, reject_failing_cases=True)
    finally:
        if old_raw is not None:
            os.environ["nnUNet_raw"] = old_raw

    rejected_img = ds_dst / "imagesTr_rejected"
    rejected_lbl = ds_dst / "labelsTr_rejected"
    assert rejected_img.is_dir()
    assert rejected_lbl.is_dir()
    updated = load_json(ds_dst / "dataset.json")
    assert updated["numTraining"] == 1
    assert len(updated["dataset"]) == 1
