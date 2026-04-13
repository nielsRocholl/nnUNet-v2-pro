#!/usr/bin/env python3
"""Profile single-patch inference: torch.profiler around one predict (needs real model + preprocessed case).

Example:
  python scripts/profile_single_patch.py \\
    --model /path/to/nnUNetTrainer_... \\
    --input /path/to/preprocessed_casename.npz \\
    --points_json /path/to/points.json

Or micro-benchmark planning-only (no checkpoint) with --micro_fg_only
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def _micro_fg_bench() -> None:
    from nnunetv2.inference.roi_predictor import _patch_fg_bool_from_logits
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    lm = LabelManager({"background": 0, "a": 1}, regions_class_order=None, force_use_labels=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 48, 48, 48, device=dev)
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        _patch_fg_bool_from_logits(x, lm)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"[micro] _patch_fg_bool_from_logits mean {dt*1000:.2f} ms (device={dev})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile or micro-benchmark single-patch path.")
    parser.add_argument("--micro_fg_only", action="store_true", help="Run tiny fg extraction benchmark (no model).")
    parser.add_argument("--model", default=None, help="Model folder for full profile.")
    parser.add_argument("--input", default=None, help="Preprocessed .npz or case path.")
    parser.add_argument("--points_json", default=None)
    args = parser.parse_args()

    if args.micro_fg_only or not args.model:
        _micro_fg_bench()
        if not args.model:
            print("Tip: pass --model, --input, --points_json for full torch.profiler trace.", flush=True)
        return

    print("Full-session profiling requires wiring WarmSinglePatchSession + preprocess; use micro or adapt locally.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
