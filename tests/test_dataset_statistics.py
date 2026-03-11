"""Test dataset statistics collection during multi-dataset preprocessing."""
import json
import os
import subprocess
import sys
from pathlib import Path

import blosc2
import numpy as np
import pytest

from nnunetv2.utilities.dataset_statistics import (
    collect_case_statistics,
    compute_size_bin_thresholds,
    get_size_bin,
    save_case_stats,
)
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


def test_collect_case_statistics(tmp_path):
    """Unit test with synthetic b2nd seg."""
    out_dir = tmp_path / "preprocessed"
    out_dir.mkdir()
    seg = np.zeros((1, 20, 20, 20), dtype=np.int8)
    seg[0, 2:7, 2:7, 2:7] = 1
    seg[0, 12:17, 12:17, 12:17] = 1
    blosc2.asarray(np.ascontiguousarray(seg), urlpath=str(out_dir / "case001_seg.b2nd"))
    blosc2.asarray(np.zeros_like(seg, dtype=np.float32), urlpath=str(out_dir / "case001.b2nd"))

    dataset_json = {"source_datasets": ["Dataset001_Mini", "Dataset002_Mini"]}
    stats = collect_case_statistics(str(out_dir), dataset_json, "Dataset999_Merged")
    assert "case001" in stats
    info = stats["case001"]
    assert info["fg_voxels"]["total"] == 250
    assert info["fg_voxels"]["min_cc"] == 125
    assert info["fg_voxels"]["max_cc"] == 125
    assert info["fg_voxels"]["mean_cc"] == 125.0
    assert info["size_bin"] == "small"


def test_collect_with_percentile_bins(tmp_path):
    out_dir = tmp_path / "preprocessed"
    out_dir.mkdir()
    seg = np.zeros((1, 20, 20, 20), dtype=np.int8)
    seg[0, 2:5, 2:5, 2:5] = 1
    seg[0, 12:15, 12:15, 12:15] = 1
    blosc2.asarray(np.ascontiguousarray(seg), urlpath=str(out_dir / "case001_seg.b2nd"))
    blosc2.asarray(np.zeros_like(seg, dtype=np.float32), urlpath=str(out_dir / "case001.b2nd"))
    for i in range(2, 15):
        s = np.zeros((1, 20, 20, 20), dtype=np.int8)
        s[0, 1:3, 1:3, 1:3] = 1
        blosc2.asarray(np.ascontiguousarray(s), urlpath=str(out_dir / f"case{i:03d}_seg.b2nd"))
        blosc2.asarray(np.zeros_like(s, dtype=np.float32), urlpath=str(out_dir / f"case{i:03d}.b2nd"))
    dataset_json = {"source_datasets": ["Dataset001_Mini", "Dataset002_Mini"]}
    size_bins_config = {"mode": "percentile", "trim_percentile": 0.025, "percentiles": [0.25, 0.5, 0.75]}
    stats = collect_case_statistics(str(out_dir), dataset_json, "Dataset999_Merged", size_bins_config=size_bins_config)
    assert "_thresholds_used" in stats
    assert "case001" in stats
    assert stats["case001"]["size_bin"] in ("tiny", "small", "medium", "large")


def test_get_size_bin():
    assert get_size_bin(0) == "background"
    assert get_size_bin(50) == "tiny"
    assert get_size_bin(500) == "small"
    assert get_size_bin(5000) == "medium"
    assert get_size_bin(50000) == "large"
    assert get_size_bin(99) == "tiny"
    assert get_size_bin(100) == "small"
    assert get_size_bin(1999) == "small"
    assert get_size_bin(2000) == "medium"
    assert get_size_bin(19999) == "medium"
    assert get_size_bin(20000) == "large"


@pytest.mark.slow
def test_dataset_statistics_collection(tmp_path):
    """Merge 2 KiTS23 mini datasets, run plan+preprocess, assert case_stats generated."""
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
    assert stats_path.exists(), f"Expected {stats_path}"

    with open(stats_path) as f:
        data = json.load(f)
    cases = {k: v for k, v in data.items() if k != "_metadata"}
    assert len(cases) >= 4
    for case_id, info in cases.items():
        assert "dataset" in info
        assert "fg_voxels" in info
        assert "size_bin" in info
        fg = info["fg_voxels"]
        assert "total" in fg and "min_cc" in fg and "max_cc" in fg and "mean_cc" in fg

    merged_json_path = raw / "Dataset999_Merged" / "dataset.json"
    with open(merged_json_path) as f:
        merged = json.load(f)
    assert "source_datasets" in merged
