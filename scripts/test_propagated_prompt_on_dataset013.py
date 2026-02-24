#!/usr/bin/env python3
"""Test propagated prompt simulation on Dataset013 Longitudinal CT samples.

Usage:
  # Using preprocessed data (after nnUNetv2_plan_and_preprocess):
  export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
  python scripts/test_propagated_prompt_on_dataset013.py

  # Or set explicit path:
  export DATASET013_PREPROCESSED_DIR=/path/to/Dataset013.../nnUNetPlans_3d_fullres
  python scripts/test_propagated_prompt_on_dataset013.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.prompt_encoding import extract_centroids_from_seg
from nnunetv2.utilities.roi_config import load_config


def main():
    parser = argparse.ArgumentParser(description="Test propagated prompt simulation on Dataset013")
    parser.add_argument(
        "--preprocessed-dir",
        default=os.environ.get(
            "DATASET013_PREPROCESSED_DIR",
            join(
                os.environ.get("nnUNet_preprocessed", ""),
                "Dataset013_Longitudinal_CT",
                "nnUNetPlans_3d_fullres",
            ),
        ),
        help="Path to preprocessed Dataset013 (nnUNetPlans_3d_fullres)",
    )
    parser.add_argument("--config", default=None, help="Path to nnunet_pro_config.json")
    parser.add_argument("--n-batches", type=int, default=20, help="Number of batches to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocessed_dir = args.preprocessed_dir
    if not os.path.isdir(preprocessed_dir):
        print(f"ERROR: Preprocessed dir not found: {preprocessed_dir}")
        print("Run nnUNetv2_plan_and_preprocess on Dataset013 first, or set DATASET013_PREPROCESSED_DIR")
        sys.exit(1)

    plans_path = join(os.path.dirname(preprocessed_dir), "nnUNetPlans.json")
    dataset_json_path = join(os.path.dirname(preprocessed_dir), "dataset.json")
    if not os.path.isfile(plans_path) or not os.path.isfile(dataset_json_path):
        print(f"ERROR: nnUNetPlans.json or dataset.json not found in {os.path.dirname(preprocessed_dir)}")
        sys.exit(1)

    config_path = args.config or join(Path(__file__).resolve().parent.parent, "tests", "fixtures", "nnunet_pro_config.json")
    cfg = load_config(config_path)

    pm = PlansManager(plans_path)
    ds_json = load_json(dataset_json_path)
    lm = pm.get_label_manager(ds_json)
    cm = pm.get_configuration("3d_fullres")
    patch_size = tuple(cm.patch_size)
    ds_cls = infer_dataset_class(preprocessed_dir)
    ds = ds_cls(preprocessed_dir)

    if len(ds.identifiers) == 0:
        print("ERROR: No preprocessed cases found")
        sys.exit(1)

    dl = nnUNetPromptAwareDataLoader(
        ds, 2, patch_size, patch_size, lm, cfg,
        oversample_foreground_percent=0.5, transforms=None,
    )

    np.random.seed(args.seed)
    n_with_prompt = 0
    n_with_offset = 0
    for i in range(args.n_batches):
        batch = next(dl)
        for b in range(batch["data"].shape[0]):
            prompt_ch = batch["data"][b, -1]
            prompt_np = prompt_ch.numpy() if hasattr(prompt_ch, "numpy") else prompt_ch
            if prompt_np.max() <= 0:
                continue
            n_with_prompt += 1
            seg = batch["target"][b, 0]
            seg_np = seg.numpy() if hasattr(seg, "numpy") else seg
            centroids = extract_centroids_from_seg(seg_np)
            prompt_peak = np.unravel_index(np.argmax(prompt_np), prompt_np.shape)
            for c in centroids:
                dist = np.sqrt(
                    (prompt_peak[0] - c[0]) ** 2
                    + (prompt_peak[1] - c[1]) ** 2
                    + (prompt_peak[2] - c[2]) ** 2
                )
                if dist > 0.5:
                    n_with_offset += 1
                    break

    print(f"Dataset013 Longitudinal CT — propagated prompt simulation test")
    print(f"  Preprocessed dir: {preprocessed_dir}")
    print(f"  Cases: {len(ds.identifiers)}")
    print(f"  Batches: {args.n_batches}")
    print(f"  Patches with prompt: {n_with_prompt}")
    print(f"  Patches with offset from centroid: {n_with_offset}")
    if n_with_prompt >= 1 and n_with_offset >= 1:
        print("  PASS: Propagated prompt simulation works as intended.")
    else:
        print("  FAIL: Expected patches with prompt and offset from centroid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
