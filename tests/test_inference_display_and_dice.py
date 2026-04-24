"""Inference Rich display and per-sample DICE."""
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join
from rich.console import Console

from nnunetv2.evaluation.evaluate_predictions import compute_dice_from_arrays
from nnunetv2.utilities.cli_display import InferenceDisplay
from nnunetv2.utilities.file_path_utilities import get_output_folder


def test_compute_dice_from_arrays_perfect():
    """pred == gt -> Dice 1.0."""
    shape = (16, 16, 16)
    seg = np.zeros(shape, dtype=np.uint8)
    seg[4:12, 4:12, 4:12] = 1
    seg[2:6, 2:6, 2:6] = 2
    pred = seg.copy()
    dice = compute_dice_from_arrays(pred, seg, [(1,), (2,)], ignore_label=None)
    assert abs(dice - 1.0) < 1e-6


def test_compute_dice_from_arrays_empty():
    """Empty foreground -> nan."""
    shape = (8, 8, 8)
    pred = np.zeros(shape, dtype=np.uint8)
    gt = np.zeros(shape, dtype=np.uint8)
    dice = compute_dice_from_arrays(pred, gt, [(1,)], ignore_label=None)
    assert np.isnan(dice)


def test_compute_dice_from_arrays_binary():
    """Binary seg, partial overlap -> 0 < Dice < 1."""
    shape = (10, 10, 10)
    gt = np.zeros(shape, dtype=np.uint8)
    gt[2:8, 2:8, 2:8] = 1
    pred = np.zeros(shape, dtype=np.uint8)
    pred[4:9, 4:9, 4:9] = 1
    dice = compute_dice_from_arrays(pred, gt, [(1,)], ignore_label=None)
    assert 0 < dice < 1


def test_compute_dice_from_arrays_multiclass():
    """Multi-class, mean over foreground labels."""
    shape = (8, 8, 8)
    gt = np.zeros(shape, dtype=np.uint8)
    gt[1:4, 1:4, 1:4] = 1
    gt[4:7, 4:7, 4:7] = 2
    pred = gt.copy()
    pred[2:3, 2:3, 2:3] = 0
    dice = compute_dice_from_arrays(pred, gt, [(1,), (2,)], ignore_label=None)
    assert 0 < dice <= 1.0


def test_inference_display_update_case_no_dice():
    """update_case without dice; no crash."""
    console = Console(file=StringIO(), force_terminal=False)
    display = InferenceDisplay("Dataset010", "3d_fullres", "cpu", 3, verbose=True)
    display.console = console
    display.progress = None
    display.update_case(1, 1.5)
    display.update_case(2, 2.0)
    assert display._dice_count == 0


def test_inference_display_update_case_with_dice():
    """update_case with dice; _dice_sum/_dice_count updated."""
    display = InferenceDisplay("Dataset010", "3d_fullres", "cpu", 3, verbose=True)
    display.console = Console(file=StringIO(), force_terminal=False)
    display.progress = None
    display.update_case(1, 1.0, dice=0.8)
    display.update_case(2, 1.0, dice=0.9)
    assert display._dice_count == 2
    assert abs(display._dice_sum - 1.7) < 1e-6


def test_inference_display_summary_includes_mean_dice():
    """When dice provided, __exit__ summary has Mean Dice row."""
    out = StringIO()
    display = InferenceDisplay("Dataset010", "3d_fullres", "cpu", 2, verbose=True)
    display.console = Console(file=out, force_terminal=False)
    display.progress = None
    display.update_case(1, 1.0, dice=0.85)
    display.update_case(2, 1.0, dice=0.75)
    display.__exit__(None, None, None)
    text = out.getvalue()
    assert "Mean Dice" in text
    assert "0.80" in text or "0.8000" in text


def test_inference_display_nan_dice_excluded():
    """NaN dice is excluded from running average."""
    display = InferenceDisplay("Dataset010", "3d_fullres", "cpu", 2, verbose=True)
    display.console = Console(file=StringIO(), force_terminal=False)
    display.progress = None
    display.update_case(1, 1.0, dice=float("nan"))
    display.update_case(2, 1.0, dice=0.9)
    assert display._dice_count == 1
    assert abs(display._dice_sum - 0.9) < 1e-6


def test_vanilla_sequential_with_display():
    """Run predict_from_files_sequential with display; no exception."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder

    preprocessed_dir = join(
        os.environ.get("nnUNet_preprocessed", ""),
        "Dataset010",
        "nnUNetPlans_3d_fullres",
    )
    if not os.path.isdir(preprocessed_dir):
        pytest.skip("Dataset010 preprocessed not found")
    model_folder = get_output_folder("Dataset010", "nnUNetTrainer", "nnUNetPlans", "3d_fullres")
    if not os.path.isdir(model_folder):
        pytest.skip("nnUNetTrainer model not found")
    raw_images = join(os.environ.get("nnUNet_raw", ""), "Dataset010", "imagesTr")
    if not os.path.isdir(raw_images):
        pytest.skip("Dataset010 imagesTr not found")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=device.type != "cpu",
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    pred.initialize_from_trained_model_folder(model_folder, (0,), "checkpoint_final.pth",)
    num_cases = len(get_identifiers_from_splitted_dataset_folder(raw_images, pred.dataset_json["file_ending"]))
    if num_cases == 0:
        pytest.skip("No cases in imagesTr")
    out_dir = join(os.environ.get("nnUNet_raw", ""), "Dataset010", "predictions_vanilla_test")
    os.makedirs(out_dir, exist_ok=True)
    with InferenceDisplay("Dataset010", "3d_fullres", str(device.type), num_cases, verbose=True) as display:
        pred.predict_from_files_sequential(
            raw_images, out_dir, save_probabilities=False, overwrite=True,
            folder_with_segs_from_prev_stage=None, display=display, labels_folder=None,
        )
    out_files = [f for f in os.listdir(out_dir) if f.endswith(".nii.gz")]
    assert len(out_files) >= 1


def test_single_patch_with_display():
    """Single-patch predict with display; no exception."""
    from nnunetv2.inference.single_patch_predict_entrypoint import predict_single_patch_entry_point

    preprocessed_dir = join(
        os.environ.get("nnUNet_preprocessed", ""),
        "Dataset010",
        "nnUNetPlans_3d_fullres",
    )
    if not os.path.isdir(preprocessed_dir):
        pytest.skip("Dataset010 preprocessed not found")
    model_folder = get_output_folder("Dataset010", "nnUNetTrainerPromptAware", "nnUNetPlans", "3d_fullres")
    if not os.path.isdir(model_folder):
        pytest.skip("nnUNetTrainerPromptAware model not found")
    raw_images = join(os.environ.get("nnUNet_raw", ""), "Dataset010", "imagesTr")
    if not os.path.isdir(raw_images):
        pytest.skip("Dataset010 imagesTr not found")

    import json
    import sys
    import tempfile
    cfg = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")
    if not os.path.isfile(cfg):
        pytest.skip("fixture config not found")
    with tempfile.TemporaryDirectory() as tmp:
        points_path = join(tmp, "points.json")
        points_json = {
            "points": [[32, 64, 64]],
            "points_space": "voxel",
            "voxel_coordinate_frame": "preprocessed",
        }
        with open(points_path, "w") as f:
            json.dump(points_json, f)
        out_dir = join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        old_argv = sys.argv
        sys.argv = [
            "nnUNetv2_predict_single_patch",
            "-i", raw_images,
            "-o", out_dir,
            "-m", model_folder,
            "--config", cfg,
            "--points_json", points_path,
            "-device", "cpu",
        ]
        try:
            predict_single_patch_entry_point()
        finally:
            sys.argv = old_argv
        out_files = [f for f in os.listdir(out_dir) if f.endswith(".nii.gz")]
        assert len(out_files) >= 1
