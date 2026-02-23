"""Test -max_patch_size: patch is capped, extra VRAM used for batch size."""
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
)
from nnunetv2.utilities.multi_dataset_merge import create_merged_dataset
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2

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

planner_cls = recursive_find_python_class(
    join(nnunetv2.__path__[0], "experiment_planning"),
    "nnUNetPlannerResEncL_torchres",
    current_module="nnunetv2.experiment_planning",
)

_, plans_id_baseline = plan_experiment_dataset(
    999, experiment_planner_class=planner_cls,
    gpu_memory_target_in_gb=24, overwrite_plans_name="nnUNetPlans_baseline", verbose=False,
)
plans_baseline = load_json(join(prep, "Dataset999_Merged", f"{plans_id_baseline}.json"))
batch_baseline = plans_baseline["configurations"]["3d_fullres"]["batch_size"]

max_ps = (96, 192, 192)
_, plans_id_capped = plan_experiment_dataset(
    999, experiment_planner_class=planner_cls,
    gpu_memory_target_in_gb=120, overwrite_plans_name="nnUNetPlans_capped", verbose=False,
    max_patch_size_in_voxels=max_ps,
)
plans_capped = load_json(join(prep, "Dataset999_Merged", f"{plans_id_capped}.json"))
cfg = plans_capped["configurations"]["3d_fullres"]
patch_capped = tuple(cfg["patch_size"])
batch_capped = cfg["batch_size"]

for i, (p, m) in enumerate(zip(patch_capped, max_ps)):
    if p > m:
        sys.exit(f"patch[{i}]={p} > max_patch_size[{i}]={m}")
if batch_capped < batch_baseline:
    sys.exit(f"batch_capped ({batch_capped}) < batch_baseline ({batch_baseline})")
"""


@pytest.mark.slow
def test_max_patch_size_caps_patch_and_increases_batch(tmp_path):
    """With max_patch_size, patch is capped and batch size uses extra VRAM."""
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
