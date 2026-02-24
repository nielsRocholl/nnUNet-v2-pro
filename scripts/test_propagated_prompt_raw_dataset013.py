#!/usr/bin/env python3
"""Test propagated prompt simulation on raw Dataset013 labelsTr-test / imagesTr-test.

Loads NIfTIs directly, builds minimal preprocessed-style data, runs dataloader.
No nnUNet preprocessing required.

Usage:
  python scripts/test_propagated_prompt_raw_dataset013.py

  # Or with custom paths:
  export nnUNet_raw=/path/to/nnUNet_raw
  python scripts/test_propagated_prompt_raw_dataset013.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batchgenerators.utilities.file_and_folder_operations import join, load_json, subfiles, write_pickle
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.prompt_encoding import extract_centroids_from_seg
from nnunetv2.utilities.roi_config import load_config

try:
    import nibabel as nib
except ImportError:
    nib = None


def _load_nifti(path):
    if nib is None:
        raise ImportError("nibabel required: pip install nibabel")
    img = nib.load(path)
    return np.asarray(img.get_fdata(), dtype=np.float32), img.affine


def _build_class_locations(seg: np.ndarray):
    """Build class_locations dict for nnUNet sampling. Format: (0, z, y, x) per row."""
    if seg.ndim == 4:
        seg = seg[0]
    fg = (seg > 0)
    if not np.any(fg):
        return {1: np.zeros((0, 4), dtype=np.int64), (-1, 1): []}
    zs, ys, xs = np.where(fg)
    n_fg = min(5000, len(zs))
    idx = np.random.choice(len(zs), n_fg, replace=False)
    locs = np.column_stack([np.zeros(n_fg), zs[idx], ys[idx], xs[idx]]).astype(np.int64)
    return {1: locs, (-1, 1): []}


def main():
    nnunet_raw = os.environ.get(
        "nnUNet_raw",
        "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/universal-lesion-segmentation/nnUNet_raw",
    )
    base = join(nnunet_raw, "Dataset013_Longitudinal_CT")
    labels_dir = join(base, "labelsTr-test")
    images_dir = join(base, "imagesTr-test")

    if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
        print(f"ERROR: labelsTr-test or imagesTr-test not found under {base}")
        sys.exit(1)

    label_files = subfiles(labels_dir, suffix=".nii.gz", join=False)
    if not label_files:
        print("ERROR: No label files in labelsTr-test")
        sys.exit(1)

    config_path = join(Path(__file__).resolve().parent.parent, "tests", "fixtures", "nnunet_pro_config.json")
    cfg = load_config(config_path)

    # Use a standard patch size for 3D
    patch_size = (64, 128, 128)
    dataset_json = load_json(join(base, "dataset.json")) if os.path.isfile(join(base, "dataset.json")) else {}
    num_classes = len(dataset_json.get("labels", {})) or 2
    if num_classes <= 1:
        num_classes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        identifiers = []
        for lf in label_files[:3]:  # Limit to 3 cases for speed
            stem = lf.replace(".nii.gz", "")
            img_f = join(images_dir, stem + "_0000.nii.gz")
            if not os.path.isfile(img_f):
                continue
            seg, _ = _load_nifti(join(labels_dir, lf))
            img, _ = _load_nifti(img_f)
            if seg.ndim == 4:
                seg = seg[0]
            if img.ndim == 4:
                img = img[0]
            seg = seg.astype(np.int16)
            if seg.max() == 0:
                continue
            data = np.stack([img], axis=0).astype(np.float32)
            seg = seg[None, ...].astype(np.int16)
            props = {
                "class_locations": _build_class_locations(seg),
                "spacing": (1.0, 1.0, 1.0),
                "shape": seg.shape,
            }
            ident = stem.replace(" ", "_")
            np.savez_compressed(join(tmpdir, ident + ".npz"), data=data, seg=seg)
            write_pickle(props, join(tmpdir, ident + ".pkl"))
            identifiers.append(ident)

        if not identifiers:
            print("ERROR: No valid cases with foreground in labelsTr-test")
            sys.exit(1)

        # Create minimal plans
        plans = {
            "configurations": {
                "3d_fullres": {
                    "patch_size": list(patch_size),
                    "spacing": [1.0, 1.0, 1.0],
                }
            }
        }
        plans_path = join(tmpdir, "nnUNetPlans.json")
        import json
        with open(plans_path, "w") as f:
            json.dump(plans, f, indent=2)
        ds_json = {"labels": {"background": 0, "lesion": 1}, "channel_names": {"0": "CT"}}
        with open(join(tmpdir, "dataset.json"), "w") as f:
            json.dump(ds_json, f, indent=2)

        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        pm = PlansManager(plans_path)
        lm = pm.get_label_manager(ds_json)
        ds = nnUNetDatasetNumpy(tmpdir, identifiers)
        dl = nnUNetPromptAwareDataLoader(
            ds, 2, patch_size, patch_size, lm, cfg,
            oversample_foreground_percent=0.0, transforms=None,
        )

        np.random.seed(42)
        n_with_prompt = 0
        n_with_offset = 0
        for _ in range(30):
            batch = next(dl)
            for b in range(batch["data"].shape[0]):
                prompt_np = batch["data"][b, -1].numpy() if hasattr(batch["data"][b, -1], "numpy") else batch["data"][b, -1]
                if prompt_np.max() <= 0:
                    continue
                n_with_prompt += 1
                seg_np = batch["target"][b, 0].numpy() if hasattr(batch["target"][b, 0], "numpy") else batch["target"][b, 0]
                centroids = extract_centroids_from_seg(seg_np)
                prompt_peak = np.unravel_index(np.argmax(prompt_np), prompt_np.shape)
                for c in centroids:
                    dist = np.sqrt(
                        (prompt_peak[0] - c[0]) ** 2 + (prompt_peak[1] - c[1]) ** 2 + (prompt_peak[2] - c[2]) ** 2
                    )
                    if dist > 0.5:
                        n_with_offset += 1
                        break

        print("Dataset013 Longitudinal CT (labelsTr-test / imagesTr-test) — propagated prompt test")
        print(f"  Cases: {len(identifiers)}")
        print(f"  Patches with prompt: {n_with_prompt}")
        print(f"  Patches with offset from centroid: {n_with_offset}")
        if n_with_prompt >= 1 and n_with_offset >= 1:
            print("  PASS: Propagated prompt simulation works on real samples.")
        else:
            print("  FAIL: Expected patches with prompt and offset.")
            sys.exit(1)


if __name__ == "__main__":
    main()
