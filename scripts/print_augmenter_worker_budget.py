#!/usr/bin/env python3
"""Print nnUNet data-augmenter process budget (same source as training). Run locally or inside the job container."""

from __future__ import annotations

import os
import sys

# Repo root on path when run as python scripts/print_augmenter_worker_budget.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


def main() -> None:
    n = get_allowed_n_proc_DA()
    n_val = max(1, n // 2)
    n_unpack = max(1, round(n // 2))
    raw = os.environ.get("nnUNet_n_proc_DA", "(unset — hostname table or default)")
    print("nnUNet_n_proc_DA env:", raw)
    print("effective N (get_allowed_n_proc_DA):", n)
    print("  training augmenter processes:", n)
    print("  validation augmenter processes:", n_val)
    print("  unpack_dataset processes (train start):", n_unpack)
    print("  approximate long-lived worker procs (excl. main):", n + n_val)


if __name__ == "__main__":
    main()
