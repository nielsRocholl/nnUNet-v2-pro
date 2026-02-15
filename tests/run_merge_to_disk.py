"""Run merge + plan + preprocess, writing output to disk for inspection. Does not remove anything."""
import os
import sys

# Must set env before importing nnunetv2
ULS_BASE = "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/universal-lesion-segmentation"
OUT_BASE = os.path.join(ULS_BASE, "merge_test_output")
RAW = os.path.join(OUT_BASE, "nnUNet_raw")
PREP = os.path.join(OUT_BASE, "nnUNet_preprocessed")
RES = os.path.join(OUT_BASE, "nnUNet_results")

os.environ["nnUNet_raw"] = RAW
os.environ["nnUNet_preprocessed"] = PREP
os.environ["nnUNet_results"] = RES

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprint_dataset,
    plan_experiment_dataset,
    preprocess_dataset,
)
from nnunetv2.utilities.multi_dataset_merge import create_merged_dataset
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

ULS_RAW = os.path.join(ULS_BASE, "nnUNet_raw")


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


def main():
    if not os.path.isdir(ULS_RAW):
        print(f"ULS nnUNet_raw not found: {ULS_RAW}")
        sys.exit(1)

    ds010 = join(ULS_RAW, "Dataset010_CECT")
    ds017 = join(ULS_RAW, "Dataset017_MSD_Liver")
    if not os.path.isdir(ds010) or not os.path.isdir(ds017):
        print("Dataset010_CECT or Dataset017_MSD_Liver not found")
        sys.exit(1)

    maybe_mkdir_p(RAW)
    maybe_mkdir_p(PREP)
    maybe_mkdir_p(RES)

    print(f"Output: {OUT_BASE}")
    print("Creating Dataset001_Mini, Dataset002_Mini (1 sample each)...")
    _create_mini_dataset(RAW, "Dataset001_Mini", ds010, 1)
    _create_mini_dataset(RAW, "Dataset002_Mini", ds017, 1)

    print("Creating merged Dataset999_Merged...")
    create_merged_dataset([1, 2], "Dataset999_Merged")

    print("Running fingerprint...")
    extract_fingerprint_dataset(999, num_processes=1, check_dataset_integrity=False, clean=True, verbose=False)
    print("Running planning...")
    _, plans_id = plan_experiment_dataset(999, verbose=False)
    print("Running preprocessing (3d_fullres)...")
    preprocess_dataset(999, plans_identifier=plans_id, configurations=("3d_fullres",), num_processes=(1,), verbose=False)

    print(f"Done. Inspect: {OUT_BASE}")


if __name__ == "__main__":
    main()
