"""Step 1: Network Input Channels — verify nnUNetTrainerPromptChannel adds prompt channel."""
import os

import numpy as np
import torch

# Set env vars before nnunetv2 imports (conftest does this via pytest_configure, but ensure for standalone run)
if "nnUNet_raw" not in os.environ:
    _base = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"
    os.environ["nnUNet_raw"] = os.path.join(_base, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(_base, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(_base, "nnUNet_results")

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

# Output dir for visual inspection (relative to project root)
STEP01_OUTPUT_DIR = "tests/outputs/step01"


def test_trainer_discoverable():
    """nnUNetTrainerPromptChannel must be findable by recursive_find_python_class."""
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        "nnUNetTrainerPromptChannel",
        "nnunetv2.training.nnUNetTrainer",
    )
    assert trainer_class is not None
    assert trainer_class.PROMPT_CHANNELS == 1


def test_build_network_adds_prompt_channel():
    """build_network_architecture(num_input_channels=1) must produce network with 2 input channels."""
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        "nnUNetTrainerPromptChannel",
        "nnunetv2.training.nnUNetTrainer",
    )
    assert trainer_class is not None

    arch = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
    kwargs = {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 320, 320],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[1, 3, 3]] + [[3, 3, 3]] * 5,
        "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "n_conv_per_stage": [2] * 6,
        "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    }
    req_import = ["conv_op", "norm_op", "dropout_op", "nonlin"]

    network = trainer_class.build_network_architecture(
        arch, kwargs, req_import,
        num_input_channels=1,
        num_output_channels=2,
        enable_deep_supervision=False,
    )

    first_conv = next(
        m for m in network.modules()
        if hasattr(m, "weight") and m.weight.dim() == 5 and m.weight.shape[1] == 2
    )
    assert first_conv.weight.shape[1] == 2, f"Expected 2 input channels, got {first_conv.weight.shape[1]}"


def test_forward_pass_2channel_input():
    """Network must accept 2-channel input (image + prompt) and produce valid output."""
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        "nnUNetTrainerPromptChannel",
        "nnunetv2.training.nnUNetTrainer",
    )
    assert trainer_class is not None

    arch = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
    kwargs = {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 320, 320],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[1, 3, 3]] + [[3, 3, 3]] * 5,
        "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "n_conv_per_stage": [2] * 6,
        "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    }
    req_import = ["conv_op", "norm_op", "dropout_op", "nonlin"]

    network = trainer_class.build_network_architecture(
        arch, kwargs, req_import,
        num_input_channels=1,
        num_output_channels=2,
        enable_deep_supervision=False,
    )

    # (batch, channels, d, h, w) — 2 channels: image + prompt
    x = torch.randn(1, 2, 32, 64, 64)
    out = network(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    assert out.shape[0] == 1 and out.shape[1] == 2
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


def _make_prompt_sphere(shape, center, radius=4):
    """Place a sphere (nnInteractive-style) in a zero array. Values in [0,1]."""
    d, h, w = shape
    zz, yy, xx = np.ogrid[:d, :h, :w]
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    sphere = np.clip(1 - dist / (radius + 1e-6), 0, 1).astype(np.float32)
    return sphere


def test_step01_visual_output():
    """Run forward pass and save NIfTIs for visual inspection (image, prompt, logits, pred)."""
    import nibabel as nib

    # Persist to tests/outputs/step01 for inspection (not tmp_path)
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(proj_root, STEP01_OUTPUT_DIR)
    maybe_mkdir_p(out_dir)

    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        "nnUNetTrainerPromptChannel",
        "nnunetv2.training.nnUNetTrainer",
    )
    assert trainer_class is not None

    arch = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
    kwargs = {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 320, 320],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[1, 3, 3]] + [[3, 3, 3]] * 5,
        "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "n_conv_per_stage": [2] * 6,
        "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    }
    req_import = ["conv_op", "norm_op", "dropout_op", "nonlin"]

    network = trainer_class.build_network_architecture(
        arch, kwargs, req_import,
        num_input_channels=1, num_output_channels=2, enable_deep_supervision=False,
    )
    network.eval()

    shape = (32, 64, 64)
    center = (16, 32, 32)
    ch0_image = np.random.randn(*shape).astype(np.float32) * 50 + 100  # CT-like
    ch1_prompt = _make_prompt_sphere(shape, center, radius=4)
    x = torch.from_numpy(np.stack([ch0_image, ch1_prompt])[None])  # (1, 2, D, H, W)

    with torch.no_grad():
        out = network(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    logits = out[0].numpy()  # (2, D, H, W)
    pred = (logits[1] > logits[0]).astype(np.uint8)  # foreground class

    def save_nii(arr, name):
        img = nib.Nifti1Image(arr, np.eye(4))
        nib.save(img, os.path.join(out_dir, name))

    save_nii(ch0_image, "input_ch0_image.nii.gz")
    save_nii(ch1_prompt, "input_ch1_prompt.nii.gz")
    save_nii(logits[1] - logits[0], "output_logits_fg_minus_bg.nii.gz")
    save_nii(pred, "output_pred.nii.gz")

    readme = os.path.join(out_dir, "README.txt")
    with open(readme, "w") as f:
        f.write("Step 1 visual outputs — inspect in CT viewer\n")
        f.write("input_ch0_image.nii.gz    synthetic CT-like image\n")
        f.write("input_ch1_prompt.nii.gz   nnInteractive-style sphere prompt [0,1]\n")
        f.write("output_logits_fg_minus_bg.nii.gz  logit difference\n")
        f.write("output_pred.nii.gz        binary prediction\n")

    assert os.path.exists(out_dir)
    assert os.path.exists(os.path.join(out_dir, "input_ch0_image.nii.gz"))
    assert os.path.exists(os.path.join(out_dir, "input_ch1_prompt.nii.gz"))
    assert os.path.exists(os.path.join(out_dir, "output_pred.nii.gz"))
