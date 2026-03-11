"""Tests for stratified batch sampling."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.stratified_data_loader import (
    nnUNetPromptAwareStratifiedDataLoader,
    nnUNetStratifiedDataLoader,
)
from nnunetv2.training.dataloading.stratified_sampling import build_strata, sample_batch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.roi_config import DEFAULT_CONFIG_PATH, load_config
from tests.conftest import KITS23_NNUNET_RAW

_TEST_SCRIPT = """
import os
import sys
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprint_dataset,
    plan_experiment_dataset,
    preprocess_dataset,
)
from nnunetv2.utilities.multi_dataset_merge import create_merged_dataset
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

def _create_mini_dataset(raw, output_name, source_path, case_ids):
    out = join(raw, output_name)
    maybe_mkdir_p(join(out, "imagesTr"))
    maybe_mkdir_p(join(out, "labelsTr"))
    ds_json = load_json(join(source_path, "dataset.json"))
    ds = get_filenames_of_train_images_and_targets(source_path, ds_json)
    subset = {}
    for cid in case_ids:
        if cid not in ds:
            sys.exit(f"Case {cid} not in source")
        entry = ds[cid]
        for img in entry["images"]:
            shutil.copy2(img, join(out, "imagesTr", os.path.basename(img)))
        shutil.copy2(entry["label"], join(out, "labelsTr", os.path.basename(entry["label"])))
        rel_label = "labelsTr/" + os.path.basename(entry["label"])
        rel_images = ["imagesTr/" + os.path.basename(i) for i in entry["images"]]
        subset[cid] = {"label": rel_label, "images": rel_images}
    out_json = {
        "channel_names": ds_json["channel_names"],
        "labels": ds_json["labels"],
        "file_ending": ds_json["file_ending"],
        "numTraining": len(subset),
        "dataset": subset,
    }
    if "overwrite_image_reader_writer" in ds_json:
        out_json["overwrite_image_reader_writer"] = ds_json["overwrite_image_reader_writer"]
    save_json(out_json, join(out, "dataset.json"), sort_keys=False)

kits = os.environ["KITS23_NNUNET_RAW"]
raw = os.environ["nnUNet_raw"]
prep = os.environ["nnUNet_preprocessed"]
_create_mini_dataset(raw, "Dataset001_Mini", kits, ["KiTS23_case_00004", "KiTS23_case_00007"])
_create_mini_dataset(raw, "Dataset002_Mini", kits, ["KiTS23_case_00015", "KiTS23_case_00016"])
create_merged_dataset([1, 2], "Dataset999_Merged")
extract_fingerprint_dataset(999, num_processes=1, check_dataset_integrity=False, clean=True, verbose=False)
_, plans_id = plan_experiment_dataset(999, verbose=False)
preprocess_dataset(999, plans_identifier=plans_id, configurations=("3d_fullres",), num_processes=(1,), verbose=False)
stats_path = join(prep, "Dataset999_Merged", "case_stats_3d_fullres.json")
if not os.path.isfile(stats_path):
    sys.exit(f"Missing {stats_path}")
"""


def test_build_strata():
    case_stats = {
        "_metadata": {"foo": "bar"},
        "case_a": {"dataset": "D1", "size_bin": "small", "fg_voxels": {}},
        "case_b": {"dataset": "D1", "size_bin": "small", "fg_voxels": {}},
        "case_c": {"dataset": "D2", "size_bin": "large", "fg_voxels": {}},
        "case_d": {"dataset": "D1", "size_bin": "large", "fg_voxels": {}},
    }
    tr_keys = ["case_a", "case_b", "case_c"]
    strata = build_strata(case_stats, tr_keys)
    assert ("D1", "small") in strata
    assert strata[("D1", "small")] == ["case_a", "case_b"]
    assert ("D2", "large") in strata
    assert strata[("D2", "large")] == ["case_c"]
    assert ("D1", "large") not in strata
    assert "_metadata" not in strata


def test_sample_batch():
    strata = {
        ("D1", "small"): ["case_a", "case_b"],
        ("D2", "large"): ["case_c"],
    }
    batch = sample_batch(strata, batch_size=4)
    assert len(batch) == 4
    valid = {"case_a", "case_b", "case_c"}
    assert all(c in valid for c in batch)


def test_sample_batch_uniform_datasets():
    strata = {
        ("D1", "small"): ["a1", "a2"],
        ("D2", "large"): ["b1", "b2"],
    }
    d1_seen = d2_seen = False
    for _ in range(20):
        batch = sample_batch(strata, batch_size=4)
        for c in batch:
            if c in ("a1", "a2"):
                d1_seen = True
            else:
                d2_seen = True
    assert d1_seen and d2_seen


def test_compute_size_bin_thresholds():
    from nnunetv2.utilities.dataset_statistics import compute_size_bin_thresholds

    stats = {
        "c1": {"fg_voxels": {"max_cc": 50}},
        "c2": {"fg_voxels": {"max_cc": 200}},
        "c3": {"fg_voxels": {"max_cc": 500}},
        "c4": {"fg_voxels": {"max_cc": 2000}},
        "c5": {"fg_voxels": {"max_cc": 5000}},
        "c6": {"fg_voxels": {"max_cc": 10000}},
        "c7": {"fg_voxels": {"max_cc": 30000}},
        "c8": {"fg_voxels": {"max_cc": 50000}},
        "c9": {"fg_voxels": {"max_cc": 100}},
        "c10": {"fg_voxels": {"max_cc": 1000}},
    }
    thresh = compute_size_bin_thresholds(stats, percentiles=(0.25, 0.5, 0.75), trim_percentile=0.025)
    assert len(thresh) == 3
    assert thresh[0] < thresh[1] < thresh[2]


def test_sample_batch_with_weights():
    strata = {
        ("D1", "small"): ["a1", "a2"],
        ("D2", "large"): ["b1", "b2"],
    }
    weights = {
        ("D1", "small"): 0.9,
        ("D2", "large"): 0.1,
    }
    d1_count = 0
    for _ in range(50):
        batch = sample_batch(strata, batch_size=4, weights=weights)
        d1_count += sum(1 for c in batch if c in ("a1", "a2"))
    assert d1_count > 120


def test_stratified_requires_case_stats(tmp_path):
    import numpy as np
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    class _MinimalDataset(nnUNetBaseDataset):
        def load_case(self, identifier):
            return (
                np.zeros((1, 32, 32, 32), dtype=np.float32),
                np.zeros((1, 32, 32, 32), dtype=np.int16),
                None,
                {},
            )

        @staticmethod
        def get_identifiers(folder):
            return ["case_a"]

        @staticmethod
        def save_case(*args, **kwargs):
            pass

    fake_path = str(tmp_path / "nonexistent_case_stats.json")
    assert not os.path.isfile(fake_path)
    dataset = _MinimalDataset(str(tmp_path), ["case_a"])
    with pytest.raises(FileNotFoundError, match="case_stats"):
        nnUNetStratifiedDataLoader(
            dataset,
            batch_size=2,
            patch_size=[32, 32, 32],
            final_patch_size=[32, 32, 32],
            label_manager=LabelManager({"background": 0, "tumor": 1}, None),
            case_stats_path=fake_path,
        )


@pytest.mark.slow
def test_nnUNetStratifiedDataLoader_integration(tmp_path):
    if not os.path.isdir(KITS23_NNUNET_RAW):
        pytest.skip(f"KiTS23 not found at {KITS23_NNUNET_RAW}")

    raw = tmp_path / "nnUNet_raw"
    prep = tmp_path / "nnUNet_preprocessed"
    res = tmp_path / "nnUNet_results"
    raw.mkdir()
    prep.mkdir()
    res.mkdir()

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw)
    env["nnUNet_preprocessed"] = str(prep)
    env["nnUNet_results"] = str(res)
    env["KITS23_NNUNET_RAW"] = KITS23_NNUNET_RAW

    result = subprocess.run(
        [sys.executable, "-c", _TEST_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    stats_path = prep / "Dataset999_Merged" / "case_stats_3d_fullres.json"
    assert stats_path.exists()
    prep_folder = prep / "Dataset999_Merged"
    plans_files = [p for p in prep_folder.glob("*.json") if "case_stats" not in p.name and "splits" not in p.name and "dataset" not in p.name]
    assert plans_files, "No plans file found"
    plans = load_json(str(plans_files[0]))
    plans_manager = PlansManager(plans)
    config = plans_manager.get_configuration("3d_fullres")
    preprocessed_folder = prep_folder / config.data_identifier
    splits = load_json(str(prep_folder / "splits_final.json"))
    tr_keys = splits["0"]["train"]
    dataset_class = infer_dataset_class(str(preprocessed_folder))
    dataset_tr = dataset_class(str(preprocessed_folder), tr_keys)
    dataset_json = load_json(str(raw / "Dataset999_Merged" / "dataset.json"))
    plans_manager = PlansManager(plans)
    label_manager = plans_manager.get_label_manager(dataset_json)
    config = plans_manager.get_configuration("3d_fullres")
    patch_size = list(config.patch_size)

    dl = nnUNetStratifiedDataLoader(
        dataset_tr,
        batch_size=2,
        patch_size=patch_size,
        final_patch_size=patch_size,
        label_manager=label_manager,
        case_stats_path=str(stats_path),
    )
    indices = dl.get_indices()
    assert len(indices) == 2
    assert all(i in tr_keys for i in indices)


@pytest.mark.slow
def test_nnUNetPromptAwareStratifiedDataLoader_integration(tmp_path):
    if not os.path.isdir(KITS23_NNUNET_RAW):
        pytest.skip(f"KiTS23 not found at {KITS23_NNUNET_RAW}")

    raw = tmp_path / "nnUNet_raw"
    prep = tmp_path / "nnUNet_preprocessed"
    res = tmp_path / "nnUNet_results"
    raw.mkdir()
    prep.mkdir()
    res.mkdir()

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw)
    env["nnUNet_preprocessed"] = str(prep)
    env["nnUNet_results"] = str(res)
    env["KITS23_NNUNET_RAW"] = KITS23_NNUNET_RAW

    result = subprocess.run(
        [sys.executable, "-c", _TEST_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    stats_path = prep / "Dataset999_Merged" / "case_stats_3d_fullres.json"
    assert stats_path.exists()
    prep_folder = prep / "Dataset999_Merged"
    plans_files = [p for p in prep_folder.glob("*.json") if "case_stats" not in p.name and "splits" not in p.name and "dataset" not in p.name]
    assert plans_files
    plans = load_json(str(plans_files[0]))
    plans_manager = PlansManager(plans)
    config = plans_manager.get_configuration("3d_fullres")
    preprocessed_folder = prep_folder / config.data_identifier
    splits = load_json(str(prep_folder / "splits_final.json"))
    tr_keys = splits["0"]["train"]
    dataset_class = infer_dataset_class(str(preprocessed_folder))
    dataset_tr = dataset_class(str(preprocessed_folder), tr_keys)
    dataset_json = load_json(str(raw / "Dataset999_Merged" / "dataset.json"))
    label_manager = plans_manager.get_label_manager(dataset_json)
    patch_size = list(config.patch_size)
    cfg = load_config(str(DEFAULT_CONFIG_PATH))

    dl = nnUNetPromptAwareStratifiedDataLoader(
        dataset_tr,
        batch_size=2,
        patch_size=patch_size,
        final_patch_size=patch_size,
        label_manager=label_manager,
        cfg=cfg,
        case_stats_path=str(stats_path),
        transforms=None,
    )
    out = dl.generate_train_batch()
    assert "data" in out
    assert "target" in out
    assert "keys" in out
    assert "mode" in out
