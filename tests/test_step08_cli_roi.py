"""Step 8: CLI contract — ROI-mode entrypoint, training --config for PromptAware."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join, load_json, load_pickle, maybe_mkdir_p, save_json

from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

STEP08_OUTPUT_DIR = "tests/outputs/step08"
FIXTURE_CONFIG = join(Path(__file__).resolve().parent, "fixtures", "nnunet_pro_config.json")
DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
PREPROCESSED_DIR_FULLRES = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans_3d_fullres",
)
PLANS_PATH = join(
    os.environ.get("nnUNet_preprocessed", join(DUMMY_BASE, "nnUNet_preprocessed")),
    "Dataset010",
    "nnUNetPlans.json",
)
RAW_IMAGES = join(os.environ.get("nnUNet_raw", join(DUMMY_BASE, "nnUNet_raw")), "Dataset010", "imagesTr")


def test_roi_cli_config_required():
    """ROI predict CLI must fail when config is missing (no --config and model folder has no nnunet_pro_config.json)."""
    from nnunetv2.inference.roi_predict_entrypoint import predict_roi_entry_point

    old_argv = sys.argv
    sys.argv = [
        "nnUNetv2_predict_roi",
        "-i", "/tmp/x",
        "-o", "/tmp/y",
        "-m", "/tmp/z",
        "--points_json", "/tmp/p.json",
    ]
    with pytest.raises(FileNotFoundError, match="Config not found"):
        predict_roi_entry_point()
    sys.argv = old_argv


def test_roi_cli_points_json_required():
    """ROI predict CLI must fail when --points_json is missing."""
    from nnunetv2.inference.roi_predict_entrypoint import predict_roi_entry_point

    old_argv = sys.argv
    sys.argv = [
        "nnUNetv2_predict_roi",
        "-i", "/tmp/x",
        "-o", "/tmp/y",
        "-m", "/tmp/z",
        "--config", FIXTURE_CONFIG,
    ]
    with pytest.raises(SystemExit):
        predict_roi_entry_point()
    sys.argv = old_argv


def test_training_cli_config_required_for_prompt_aware():
    """get_trainer_from_args must raise when nnUNetTrainerPromptAware and no config_path."""
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    with pytest.raises(ValueError, match="--config is required"):
        get_trainer_from_args(
            "Dataset010",
            "3d_fullres",
            0,
            trainer_name="nnUNetTrainerPromptAware",
            config_path=None,
        )


def test_training_accepts_config_for_prompt_aware():
    """get_trainer_from_args accepts config_path for nnUNetTrainerPromptAware."""
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    trainer = get_trainer_from_args(
        "Dataset010",
        "3d_fullres",
        0,
        trainer_name="nnUNetTrainerPromptAware",
        config_path=FIXTURE_CONFIG,
    )
    assert trainer.roi_cfg is not None
    assert trainer.roi_cfg.prompt.point_radius_vox == 5


def _make_temp_model_folder(tmpdir):
    """Create minimal model folder for ROI predict (untrained weights)."""
    import torch
    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPromptAware import nnUNetTrainerPromptAware
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    plans = load_json(PLANS_PATH)
    ds_json = load_json(join(os.path.dirname(PREPROCESSED_DIR_FULLRES), "dataset.json"))
    pm = PlansManager(PLANS_PATH)
    cm = pm.get_configuration("3d_fullres")
    lm = pm.get_label_manager(ds_json)
    num_in = determine_num_input_channels(pm, cm, ds_json)
    network = nnUNetTrainerPromptAware.build_network_architecture(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        num_in,
        lm.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    ckpt = {
        "network_weights": network.state_dict(),
        "trainer_name": "nnUNetTrainerPromptAware",
        "init_args": {"configuration": "3d_fullres"},
        "inference_allowed_mirroring_axes": None,
    }
    model_dir = join(tmpdir, "model")
    maybe_mkdir_p(join(model_dir, "fold_0"))
    save_json(ds_json, join(model_dir, "dataset.json"))
    save_json(plans, join(model_dir, "plans.json"))
    torch.save(ckpt, join(model_dir, "fold_0", "checkpoint_final.pth"))
    return model_dir


def test_roi_cli_end_to_end():
    """Run ROI predict CLI on real data with temp model. Asserts output exists."""
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    if not os.path.isdir(RAW_IMAGES):
        pytest.skip("Raw images folder not found")
    try:
        import blosc2
    except ImportError:
        pytest.skip("blosc2 required for preprocessed data")

    ids = [
        i[:-5]
        for i in os.listdir(PREPROCESSED_DIR_FULLRES)
        if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")
    ]
    if not ids:
        pytest.skip("No preprocessed data")
    ident = ids[0]
    data = np.array(blosc2.open(join(PREPROCESSED_DIR_FULLRES, ident + ".b2nd"), mode="r"))
    props = load_pickle(join(PREPROCESSED_DIR_FULLRES, ident + ".pkl"))
    shape = data.shape[1:]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = _make_temp_model_folder(tmpdir)
        points_path = join(tmpdir, "points.json")
        with open(points_path, "w") as f:
            json.dump({"points": [list(center)], "points_space": "voxel"}, f)
        out_dir = join(tmpdir, "out")
        maybe_mkdir_p(out_dir)

        old_argv = sys.argv
        sys.argv = [
            "nnUNetv2_predict_roi",
            "-i", RAW_IMAGES,
            "-o", out_dir,
            "-m", model_dir,
            "--config", FIXTURE_CONFIG,
            "--points_json", points_path,
            "-f", "0",
            "-device", "cpu",
            "-npp", "1",
            "-nps", "1",
        ]
        try:
            from nnunetv2.inference.roi_predict_entrypoint import predict_roi_entry_point
            predict_roi_entry_point()
        finally:
            sys.argv = old_argv

        file_ending = load_json(join(model_dir, "dataset.json"))["file_ending"]
        out_file = join(out_dir, ident + file_ending)
        assert os.path.isfile(out_file), f"Expected output {out_file}"


def test_step08_visual_output():
    """Save NIfTIs for CT viewer: prediction from ROI CLI."""
    if not os.path.isfile(PLANS_PATH):
        pytest.skip("nnUNetPlans.json not found")
    if not os.path.isdir(RAW_IMAGES):
        pytest.skip("Raw images folder not found")
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
    data = np.array(blosc2.open(join(PREPROCESSED_DIR_FULLRES, ident + ".b2nd"), mode="r"))
    shape = data.shape[1:]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = join(proj_root, STEP08_OUTPUT_DIR)
    maybe_mkdir_p(out_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = _make_temp_model_folder(tmpdir)
        points_path = join(tmpdir, "points.json")
        with open(points_path, "w") as f:
            json.dump({"points": [list(center)], "points_space": "voxel"}, f)
        roi_out = join(tmpdir, "roi_out")
        maybe_mkdir_p(roi_out)

        old_argv = sys.argv
        sys.argv = [
            "nnUNetv2_predict_roi",
            "-i", RAW_IMAGES,
            "-o", roi_out,
            "-m", model_dir,
            "--config", FIXTURE_CONFIG,
            "--points_json", points_path,
            "-f", "0",
            "-device", "cpu",
            "-npp", "1",
            "-nps", "1",
        ]
        try:
            from nnunetv2.inference.roi_predict_entrypoint import predict_roi_entry_point
            predict_roi_entry_point()
        finally:
            sys.argv = old_argv

        file_ending = load_json(join(model_dir, "dataset.json"))["file_ending"]
        pred_path = join(roi_out, ident + file_ending)
        if os.path.isfile(pred_path):
            import shutil
            shutil.copy(pred_path, join(out_dir, "prediction.nii.gz"))
        with open(join(out_dir, "points_used.json"), "w") as f:
            json.dump({"points": [list(center)], "points_space": "voxel"}, f, indent=2)

    with open(join(out_dir, "README.txt"), "w") as f:
        f.write("Step 8 visual outputs — ROI-mode CLI\n\n")
        f.write("prediction.nii.gz  — overlay on original CT in imagesTr\n")
        f.write("points_used.json   — centroid point used for prompt\n")

    assert os.path.exists(join(out_dir, "README.txt"))
