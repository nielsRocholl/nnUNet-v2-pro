"""Offline cc3d centroid + bbox extraction for prompt-aware training. Writes {id}_centroids.json per case."""
import json
import multiprocessing
import os
from typing import Any, Dict, List

import blosc2
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import isfile, join
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.large_lesion_sampling import get_lesion_bboxes_zyx
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.prompt_encoding import extract_centroids_from_seg


def compute_centroids_dict(seg: np.ndarray) -> Dict[str, Any]:
    cents = extract_centroids_from_seg(seg)
    boxes = get_lesion_bboxes_zyx(seg)
    return {
        "centroids_zyx": [list(c) for c in cents],
        "bboxes_zyx": [list(b) for b in boxes],
    }


def _process_one(folder: str, identifier: str, resume: bool) -> None:
    out_json = join(folder, identifier + "_centroids.json")
    if resume and isfile(out_json):
        return
    seg_f = join(folder, identifier + "_seg.b2nd")
    if not isfile(seg_f):
        return
    blosc2.set_nthreads(1)
    mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
    seg = blosc2.open(urlpath=seg_f, mode="r", dparams={"nthreads": 1}, **mmap_kwargs)
    seg_arr = np.asarray(seg)
    payload = compute_centroids_dict(seg_arr)
    del seg_arr
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def precompute_centroids_for_folder(folder: str, num_processes: int, resume: bool = False) -> None:
    if not os.path.isdir(folder):
        return
    suf = "_seg.b2nd"
    ids: List[str] = sorted(i[: -len(suf)] for i in os.listdir(folder) if i.endswith(suf))
    if not ids:
        return
    if num_processes <= 1:
        for i in ids:
            _process_one(folder, i, resume)
        return
    args = [(folder, i, resume) for i in ids]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        pool.starmap(_process_one, args)


def precompute_centroids_entry() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Write {case}_centroids.json next to each _seg.b2nd in a preprocessed config folder."
    )
    parser.add_argument("-d", type=int, required=True, help="Dataset ID (e.g. 999)")
    parser.add_argument("-plans_name", default="nnUNetPlans", help="Plans JSON name without .json")
    parser.add_argument("-c", type=str, required=True, help="Configuration (e.g. 3d_fullres)")
    parser.add_argument("-np", type=int, default=8, help="Worker processes")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases that already have _centroids.json",
    )
    args = parser.parse_args()

    dataset_name = convert_id_to_dataset_name(args.d)
    plans_file = join(nnUNet_preprocessed, dataset_name, args.plans_name + ".json")
    pm = PlansManager(plans_file)
    cm = pm.get_configuration(args.c)
    folder = join(nnUNet_preprocessed, dataset_name, cm.data_identifier)
    precompute_centroids_for_folder(folder, max(1, args.np), resume=args.resume)
