#!/usr/bin/env python3
"""
Update patch_size and batch_size in nnUNet plans WITHOUT re-running preprocessing.

Use when you want different GPU targets than used during plan_and_preprocess:
- patch_size: fit inference on smaller GPU (e.g. 20GB)
- batch_size: maximize for training GPU (e.g. 140GB H200)

Preprocessed data is patch-size agnostic; training crops at runtime.

Usage:
  python scripts/update_plans_patch_batch.py \\
    --preprocessed-dir /path/to/nnUNet_preprocessed/Dataset999_Merged \\
    --plans-name nnUNetResEncUNetLPlans_h200 \\
    --config 3d_fullres \\
    --patch-size 96 128 128 \\
    --batch-size 8

Divisibility: patch_size must be divisible by the network's shape_must_be_divisible_by.
For ResEncUNet 7-stage: typically [32, 64, 64] (D, H, W).
"""
import argparse
import os
import sys

from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preprocessed-dir", required=True, help="e.g. nnUNet_preprocessed/Dataset999_Merged")
    parser.add_argument("--plans-name", default="nnUNetResEncUNetLPlans_h200", help="Plans identifier (no .json)")
    parser.add_argument("--config", default="3d_fullres", help="Configuration to modify")
    parser.add_argument("--patch-size", type=int, nargs=3, metavar=("D", "H", "W"), help="e.g. 96 128 128 for ~20GB")
    parser.add_argument("--batch-size", type=int, help="e.g. 8 for 140GB H200")
    args = parser.parse_args()

    plans_path = join(args.preprocessed_dir, args.plans_name + ".json")
    if not os.path.isfile(plans_path):
        print(f"ERROR: Plans file not found: {plans_path}")
        sys.exit(1)

    plans = load_json(plans_path)
    if "configurations" not in plans or args.config not in plans["configurations"]:
        print(f"ERROR: Configuration '{args.config}' not found in plans")
        sys.exit(1)

    cfg = plans["configurations"][args.config]
    changed = []

    if args.patch_size:
        div = [32, 64, 64]
        for i, (p, d) in enumerate(zip(args.patch_size, div)):
            if p % d != 0:
                print(f"ERROR: patch_size[{i}]={p} must be divisible by {d}")
                sys.exit(1)
        old_ps = cfg.get("patch_size")
        cfg["patch_size"] = list(args.patch_size)
        changed.append(f"patch_size: {old_ps} -> {cfg['patch_size']}")

    if args.batch_size is not None:
        if args.batch_size < 1:
            print("ERROR: batch_size must be >= 1")
            sys.exit(1)
        old_bs = cfg.get("batch_size")
        cfg["batch_size"] = args.batch_size
        changed.append(f"batch_size: {old_bs} -> {cfg['batch_size']}")

    if not changed:
        print("No changes requested. Use --patch-size and/or --batch-size")
        sys.exit(0)

    save_json(plans, plans_path, sort_keys=False)
    print(f"Updated {plans_path}:")
    for c in changed:
        print(f"  {c}")


if __name__ == "__main__":
    main()
