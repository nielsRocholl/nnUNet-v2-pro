"""E2E: train (overfit) + ROI predict on Dataset010. Slow — run manually with pytest -m e2e."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join, load_json, load_pickle, maybe_mkdir_p

from nnunetv2.run.run_training import run_training
from nnunetv2.utilities.file_path_utilities import get_output_folder

FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")
DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
PREPROCESSED_DIR_FULLRES = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_fullres",
)
RAW_IMAGES = join(os.environ.get("nnUNet_raw", join(DUMMY_BASE, "nnUNet_raw")), "Dataset010", "imagesTr")


@pytest.mark.e2e
@pytest.mark.slow
def test_e2e_train_overfit_and_roi_predict():
    """Full pipeline: train 3 epochs (overfit) on 1 sample, then ROI predict."""
    if not os.path.isdir(PREPROCESSED_DIR_FULLRES):
        pytest.skip("Dataset010 fullres not found")
    try:
        import blosc2
    except ImportError:
        pytest.skip("blosc2 required")
    ids = [
        i[:-5]
        for i in os.listdir(PREPROCESSED_DIR_FULLRES)
        if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")
    ]
    if not ids:
        pytest.skip("No preprocessed data")
    ident = ids[0]
    seg = np.array(blosc2.open(join(PREPROCESSED_DIR_FULLRES, ident + "_seg.b2nd"), mode="r"))
    shape = seg.shape[1:]
    import cc3d
    seg_bin = (np.asarray(seg)[0] > 0).astype(np.uint8)
    labels = cc3d.connected_components(seg_bin)
    stats = cc3d.statistics(labels, no_slice_conversion=True)
    centroids = stats["centroids"][1:]
    center = (
        (int(np.round(centroids[0][0])), int(np.round(centroids[0][1])), int(np.round(centroids[0][2])))
        if len(centroids) > 0
        else (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    )

    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    os.environ.setdefault("nnUNet_n_proc_DA", "0")

    run_training(
        "Dataset010",
        "3d_fullres",
        "all",
        trainer_class_name="nnUNetTrainerPromptAware",
        config_path=FIXTURE_CONFIG,
        num_epochs=3,
        disable_checkpointing=True,
        device=device,
    )

    model_folder = get_output_folder(
        "Dataset010", "nnUNetTrainerPromptAware", "nnUNetPlans", "3d_fullres", fold=None
    )
    out_dir = join(os.environ["nnUNet_raw"], "Dataset010", "predictions_roi_e2e")
    maybe_mkdir_p(out_dir)
    points_path = join(out_dir, "points.json")
    with open(points_path, "w") as f:
        json.dump({"points": [list(center)], "points_space": "voxel"}, f)

    old_argv = sys.argv
    sys.argv = [
        "nnUNetv2_predict_roi",
        "-i", RAW_IMAGES,
        "-o", out_dir,
        "-m", model_folder,
        "--config", FIXTURE_CONFIG,
        "--points_json", points_path,
        "-f", "0",
        "-device", "cpu" if device.type == "cpu" else "cuda",
        "-npp", "1",
        "-nps", "1",
    ]
    try:
        from nnunetv2.inference.roi_predict_entrypoint import predict_roi_entry_point
        predict_roi_entry_point()
    finally:
        sys.argv = old_argv

    file_ending = load_json(join(model_folder, "dataset.json"))["file_ending"]
    pred_file = join(out_dir, ident + file_ending)
    assert os.path.isfile(pred_file), f"Expected {pred_file}"
