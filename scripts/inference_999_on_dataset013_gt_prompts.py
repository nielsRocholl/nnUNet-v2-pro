#!/usr/bin/env python3
"""Run prompt-aware ROI inference on Dataset013 test samples using GT lesion centroids as prompts.

Outputs per-sample Dice. Uses nnUNetv2_predict_roi machinery programmatically (per-sample points).
CPU device by default for M4 Mac compatibility.

Usage:
  python scripts/inference_999_on_dataset013_gt_prompts.py

  # Or with custom paths:
  python scripts/inference_999_on_dataset013_gt_prompts.py \\
    --model /path/to/model_folder \\
    --images /path/to/nnUNet_raw/Dataset013_Longitudinal_CT/imagesTr-test \\
    --labels /path/to/nnUNet_raw/Dataset013_Longitudinal_CT/labelsTr-test \\
    --output ./predictions
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isfile,
    maybe_mkdir_p,
    save_json,
    subdirs,
    subfiles,
)
from nnunetv2.evaluation.evaluate_predictions import compute_metrics
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.roi_predictor import nnUNetROIPredictor
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.prompt_encoding import extract_centroids_from_seg
from nnunetv2.utilities.roi_config import load_config
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder


def _resolve_model_folder(path: str) -> str:
    """Resolve to trainer output folder (contains fold_0, dataset.json, nnunet_pro_config.json)."""
    if isfile(join(path, "nnunet_pro_config.json")) and isfile(join(path, "dataset.json")):
        return path
    for sub in subdirs(path, join=True):
        if isfile(join(sub, "nnunet_pro_config.json")) and isfile(join(sub, "dataset.json")):
            return sub
    raise FileNotFoundError(
        f"Model folder must contain nnunet_pro_config.json and dataset.json. "
        f"Not found in {path} or its subdirs."
    )


def _build_case_list(images_dir: str, labels_dir: str, file_ending: str):
    """Return [(caseid, image_files, label_file), ...] for cases with both image and label."""
    identifiers = get_identifiers_from_splitted_dataset_folder(images_dir, file_ending)
    label_files = {f.replace(file_ending, ""): join(labels_dir, f) for f in subfiles(labels_dir, suffix=file_ending, join=False)}
    cases = []
    for ident in identifiers:
        label_path = label_files.get(ident)
        if label_path is None or not os.path.isfile(label_path):
            print(f"  Skipping {ident}: no matching label")
            continue
        image_files = [join(images_dir, f) for f in subfiles(images_dir, suffix=file_ending, join=False) if f.startswith(ident + "_")]
        if not image_files:
            print(f"  Skipping {ident}: no image files")
            continue
        cases.append((ident, sorted(image_files), label_path))
    return cases


def main():
    parser = argparse.ArgumentParser(description="Prompt-aware inference with GT centroids, compute Dice per sample.")
    nnunet_raw = os.environ.get("nnUNet_raw", "")
    default_base = join(nnunet_raw, "Dataset013_Longitudinal_CT") if nnunet_raw else ""
    parser.add_argument("--model", default="/Users/nielsrocholl/Downloads/Dataset999_Merged", help="Model folder (trainer output)")
    parser.add_argument("--images", default=join(default_base, "imagesTr-test") if default_base else "", help="Images folder")
    parser.add_argument("--labels", default=join(default_base, "labelsTr-test") if default_base else "", help="Labels folder")
    parser.add_argument("--output", default="./predictions_dataset013_gt_prompts", help="Output folder")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device (cpu for M4 Mac)")
    parser.add_argument("-f", type=int, default=0, help="Fold to use")
    parser.add_argument("-chk", default="checkpoint_best.pth", help="Checkpoint name (default: checkpoint_best.pth)")
    args = parser.parse_args()

    if not args.images or not args.labels:
        parser.error("--images and --labels required. Set nnUNet_raw or pass paths explicitly.")

    model_folder = _resolve_model_folder(args.model)
    maybe_mkdir_p(args.output)

    device = torch.device(args.device)
    if device.type == "cpu":
        import multiprocessing as mp
        torch.set_num_threads(mp.cpu_count())

    config_path = join(model_folder, "nnunet_pro_config.json")
    cfg = load_config(config_path)

    pred = nnUNetROIPredictor(
        tile_step_size=cfg.inference.tile_step_size,
        use_gaussian=True,
        use_mirroring=not cfg.inference.disable_tta_default,
        perform_everything_on_device=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    pred.initialize_from_trained_model_folder(model_folder, use_folds=(args.f,), checkpoint_name=args.chk)

    file_ending = pred.dataset_json["file_ending"]
    preprocessor = pred.configuration_manager.preprocessor_class(verbose=False)
    cases = _build_case_list(args.images, args.labels, file_ending)
    if not cases:
        print("No cases found. Check --images and --labels paths.")
        sys.exit(1)

    label_manager = pred.plans_manager.get_label_manager(pred.dataset_json)
    regions = label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels
    rw = pred.plans_manager.image_reader_writer_class()

    results = []
    for caseid, image_files, label_file in cases:
        data, seg, props = preprocessor.run_case(
            image_files, label_file, pred.plans_manager, pred.configuration_manager, pred.dataset_json
        )
        data = torch.from_numpy(data).float()
        shape = tuple(data.shape[1:])
        centroids = extract_centroids_from_seg(np.asarray(seg[0]))
        if not centroids:
            centroids = [(shape[0] // 2, shape[1] // 2, shape[2] // 2)]

        logits = pred.predict_logits_roi_mode(data, centroids, props, cfg)
        ofile = join(args.output, caseid)
        export_prediction_from_logits(
            logits.cpu().numpy(),
            props,
            pred.configuration_manager,
            pred.plans_manager,
            pred.dataset_json,
            ofile,
            save_probabilities=False,
        )
        pred_file = ofile + file_ending
        metrics = compute_metrics(label_file, pred_file, rw, regions, ignore_label=label_manager.ignore_label)
        dice = np.nanmean([metrics["metrics"][r]["Dice"] for r in regions])
        results.append((caseid, dice))
        print(f"{caseid}: Dice={dice:.4f}")

    if results:
        mean_dice = np.nanmean([r[1] for r in results])
        print(f"Mean Dice: {mean_dice:.4f}")
        save_json(
            {"metric_per_case": [{"reference_file": r[0], "Dice": r[1]} for r in results], "mean_Dice": float(mean_dice)},
            join(args.output, "summary.json"),
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
