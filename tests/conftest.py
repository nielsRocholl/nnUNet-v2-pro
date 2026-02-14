"""Pytest configuration. Sets nnUNet env vars for tests that need dummy data."""
import os


# Path to dummy datasets (set before importing nnunetv2.paths)
NNUNET_DUMMY_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets"

# Path to ULS nnUNet_raw (19 datasets) for merge tests
ULS_NNUNET_RAW = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/universal-lesion-segmentation/nnUNet_raw"


def _set_nnunet_paths():
    os.environ["nnUNet_raw"] = os.path.join(NNUNET_DUMMY_BASE, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(NNUNET_DUMMY_BASE, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(NNUNET_DUMMY_BASE, "nnUNet_results")


def pytest_configure(config):
    _set_nnunet_paths()
    config.addinivalue_line("markers", "e2e: end-to-end test (train + predict). Slow.")
    config.addinivalue_line("markers", "slow: slow test.")
