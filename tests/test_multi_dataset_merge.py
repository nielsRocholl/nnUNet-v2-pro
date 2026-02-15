"""Test multi-dataset merge: create virtual merged dataset, run fingerprint/plan/preprocess."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import ULS_NNUNET_RAW

_TEST_SCRIPT = """
import os
import sys
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprint_dataset,
    plan_experiment_dataset,
    preprocess_dataset,
)
from nnunetv2.utilities.multi_dataset_merge import create_merged_dataset
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

def _create_mini_dataset(temp_raw, output_name, source_path, num_samples=1):
    out_dir = join(temp_raw, output_name)
    maybe_mkdir_p(out_dir)
    ds_json = load_json(join(source_path, "dataset.json"))
    ds = get_filenames_of_train_images_and_targets(source_path, ds_json)
    keys = list(ds.keys())[:num_samples]
    subset = {k: {"label": ds[k]["label"], "images": ds[k]["images"]} for k in keys}
    out_json = {
        "channel_names": ds_json["channel_names"],
        "labels": ds_json["labels"],
        "file_ending": ds_json["file_ending"],
        "numTraining": len(subset),
        "dataset": subset,
    }
    if "overwrite_image_reader_writer" in ds_json:
        out_json["overwrite_image_reader_writer"] = ds_json["overwrite_image_reader_writer"]
    save_json(out_json, join(out_dir, "dataset.json"), sort_keys=False)

uls = os.environ["ULS_NNUNET_RAW"]
raw = os.environ["nnUNet_raw"]
prep = os.environ["nnUNet_preprocessed"]
ds010 = join(uls, "Dataset010_CECT")
ds017 = join(uls, "Dataset017_MSD_Liver")
_create_mini_dataset(raw, "Dataset001_Mini", ds010, 1)
_create_mini_dataset(raw, "Dataset002_Mini", ds017, 1)
create_merged_dataset([1, 2], "Dataset999_Merged")
extract_fingerprint_dataset(999, num_processes=1, check_dataset_integrity=False, clean=True, verbose=False)
_, plans_id = plan_experiment_dataset(999, verbose=False)
preprocess_dataset(999, plans_identifier=plans_id, configurations=("3d_fullres",), num_processes=(1,), verbose=False)
fp = join(prep, "Dataset999_Merged", "dataset_fingerprint.json")
plans = join(prep, "Dataset999_Merged", "nnUNetPlans.json")
fullres = join(prep, "Dataset999_Merged", "nnUNetPlans_3d_fullres")
if not os.path.isfile(fp):
    sys.exit(f"Missing {fp}")
if not os.path.isfile(plans):
    sys.exit(f"Missing {plans}")
if not os.path.isdir(fullres):
    sys.exit(f"Missing dir {fullres}")
b2nd = [f for f in os.listdir(fullres) if f.endswith(".b2nd") and not f.endswith("_seg.b2nd")]
if len(b2nd) < 2:
    sys.exit(f"Expected >=2 .b2nd files, got {len(b2nd)}")
"""


@pytest.mark.slow
def test_merge_plan_and_preprocess(tmp_path):
    """Merge 2 minimal ULS datasets (1 sample each), run fingerprint/plan/preprocess in subprocess."""
    if not os.path.isdir(ULS_NNUNET_RAW):
        pytest.skip(f"ULS data not found at {ULS_NNUNET_RAW}")

    raw = tmp_path / "nnUNet_raw"
    prep = tmp_path / "nnUNet_preprocessed"
    res = tmp_path / "nnUNet_results"
    raw.mkdir()
    prep.mkdir()
    res.mkdir()

    ds010 = os.path.join(ULS_NNUNET_RAW, "Dataset010_CECT")
    ds017 = os.path.join(ULS_NNUNET_RAW, "Dataset017_MSD_Liver")
    if not os.path.isdir(ds010) or not os.path.isdir(ds017):
        pytest.skip("Dataset010_CECT or Dataset017_MSD_Liver not found")

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw)
    env["nnUNet_preprocessed"] = str(prep)
    env["nnUNet_results"] = str(res)
    env["ULS_NNUNET_RAW"] = ULS_NNUNET_RAW

    result = subprocess.run(
        [sys.executable, "-c", _TEST_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
