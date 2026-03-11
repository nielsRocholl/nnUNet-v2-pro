#!/usr/bin/env python3
"""Extract patches around each GT connected component, compute patch-level Dice, save patches.
Patches are centered on GT CCs, resampled to isotropic for visualization.

Usage:
    python scripts/patch_level_dice_analysis.py

Inputs (configured in __main__):
    - images_dir: Directory containing input images (format: {case_id}_0000.nii.gz)
    - labels_dir: Directory containing ground truth labels (format: {case_id}.nii.gz)
    - predictions_dir: Directory containing predictions (format: {case_id}.nii.gz)

Outputs:
    - output_dir/patches/image/: Image patches (resampled to 1mm isotropic)
    - output_dir/patches/gt/: Ground truth patches (resampled to 1mm isotropic)
    - output_dir/patches/pred/: Prediction patches (resampled to 1mm isotropic)
    - Console output: Patch-level Dice scores per case and overall statistics

Parameters:
    - padding_voxels: Padding around each GT connected component (default: 40)
    - target_spacing_mm: Target spacing for resampling (default: 1.0 mm)
    - file_ending: File extension (default: ".nii.gz")

For each GT connected component:
    1. Extract a patch centered on the CC with padding
    2. Compute Dice between pred and GT within the patch
    3. Resample patches to isotropic spacing for visualization
    4. Save patches with matching filenames in separate folders
"""
import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.evaluation.evaluate_predictions import compute_dice_from_arrays


def get_cc_bboxes(seg: np.ndarray, label: int = 1, min_voxels: int = 10):
    """Return list of (min_coords, max_coords) for each connected component."""
    mask = (seg == label).astype(np.uint8)
    labeled, n_cc = ndimage.label(mask)
    bboxes = []
    for i in range(1, n_cc + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) < min_voxels:
            continue
        bboxes.append((tuple(coords.min(axis=0)), tuple(coords.max(axis=0))))
    return bboxes


def pad_bbox(min_c, max_c, shape, padding_voxels: int = 30):
    """Expand bbox by padding, clamp to volume."""
    min_p = [max(0, min_c[i] - padding_voxels) for i in range(3)]
    max_p = [min(shape[i] - 1, max_c[i] + padding_voxels) for i in range(3)]
    return min_p, max_p


def extract_patch(arr: np.ndarray, min_c, max_c):
    return arr[min_c[0] : max_c[0] + 1, min_c[1] : max_c[1] + 1, min_c[2] : max_c[2] + 1]


def resample_to_isotropic(arr: np.ndarray, spacing_xyz: tuple, target_spacing: float = 1.0, is_seg: bool = False):
    """Resample 3D array to isotropic spacing. arr is (z,y,x), spacing_xyz is (x,y,z)."""
    # sitk uses (x,y,z), array is (z,y,x) -> spacing for array axes: [spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]]
    arr = np.asarray(arr)
    itk = sitk.GetImageFromArray(arr.astype(np.float32 if not is_seg else np.uint8))
    itk.SetSpacing([float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])])
    target_size = [
        int(round(arr.shape[2] * spacing_xyz[0] / target_spacing)),
        int(round(arr.shape[1] * spacing_xyz[1] / target_spacing)),
        int(round(arr.shape[0] * spacing_xyz[2] / target_spacing)),
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing([target_spacing] * 3)
    resampler.SetOutputOrigin(itk.GetOrigin())
    resampler.SetOutputDirection(itk.GetDirection())
    resampler.SetDefaultPixelValue(0 if is_seg else -1024)
    if is_seg:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    out = resampler.Execute(itk)
    return np.asarray(sitk.GetArrayFromImage(out))


def run(
    images_dir: str,
    labels_dir: str,
    predictions_dir: str,
    output_dir: str,
    file_ending: str = ".nii.gz",
    padding_voxels: int = 40,
    target_spacing_mm: float = 1.0,
):
    rw = SimpleITKIO()
    os.makedirs(output_dir, exist_ok=True)
    for sub in ("image", "gt", "pred"):
        os.makedirs(join(output_dir, "patches", sub), exist_ok=True)

    pred_files = subfiles(predictions_dir, suffix=file_ending, join=False)
    case_ids = [f[: -len(file_ending)] for f in pred_files]

    all_dices = []
    results = []

    for case_id in case_ids:
        img_path = join(images_dir, case_id + "_0000" + file_ending)
        label_path = join(labels_dir, case_id + file_ending)
        pred_path = join(predictions_dir, case_id + file_ending)

        if not os.path.isfile(img_path) or not os.path.isfile(label_path) or not os.path.isfile(pred_path):
            continue

        img, img_props = rw.read_images((img_path,))
        label, _ = rw.read_seg(label_path)
        pred, _ = rw.read_seg(pred_path)

        img = np.squeeze(img)
        label = np.squeeze(label)
        pred = np.squeeze(pred)

        spacing_xyz = img_props.get("sitk_stuff", {}).get("spacing", (1, 1, 1))
        if hasattr(spacing_xyz, "__iter__") and len(spacing_xyz) == 3:
            spacing_xyz = tuple(float(s) for s in spacing_xyz)
        else:
            spacing_xyz = (1.0, 1.0, 1.0)

        bboxes = get_cc_bboxes(label, label=1)
        if not bboxes:
            continue

        case_dices = []
        for cc_idx, (min_c, max_c) in enumerate(bboxes):
            min_p, max_p = pad_bbox(min_c, max_c, label.shape, padding_voxels)

            img_patch = extract_patch(img, min_p, max_p)
            gt_patch = extract_patch(label, min_p, max_p)
            pred_patch = extract_patch(pred, min_p, max_p)

            dice = compute_dice_from_arrays(pred_patch, gt_patch, [(1,)], None)
            if not np.isnan(dice):
                case_dices.append(dice)
                all_dices.append(dice)

            # Resample to isotropic for consistent visualization
            patch_spacing = (
                spacing_xyz[0],
                spacing_xyz[1],
                spacing_xyz[2],
            )
            img_iso = resample_to_isotropic(img_patch, patch_spacing, target_spacing_mm, is_seg=False)
            gt_iso = resample_to_isotropic(gt_patch, patch_spacing, target_spacing_mm, is_seg=True)
            pred_iso = resample_to_isotropic(pred_patch, patch_spacing, target_spacing_mm, is_seg=True)

            safe_id = case_id.replace("/", "_").replace(" ", "_")
            fname = f"{safe_id}_cc{cc_idx}.nii.gz"
            for name, arr in [("image", img_iso), ("gt", gt_iso), ("pred", pred_iso)]:
                out_path = join(output_dir, "patches", name, fname)
                itk = sitk.GetImageFromArray(arr.astype(np.float32 if name == "image" else np.uint8))
                itk.SetSpacing([target_spacing_mm] * 3)
                sitk.WriteImage(itk, out_path, True)

        mean_case = np.mean(case_dices) if case_dices else np.nan
        results.append((case_id, len(bboxes), case_dices, mean_case))
        print(f"{case_id}: {len(bboxes)} CCs, patch Dice: {[f'{d:.3f}' for d in case_dices]}, mean={mean_case:.3f}")

    print("\n" + "=" * 60)
    print(f"Total patches: {len(all_dices)}")
    print(f"Mean patch-level Dice: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}" if all_dices else "No patches")
    print(f"Patches saved to {join(output_dir, 'patches')} (image/, gt/, pred/)")

    return results


if __name__ == "__main__":
    run(
        images_dir="/nnunet_data/nnUNet_raw/Dataset999_20LongitudinalCT_dummy/imagesTr",
        labels_dir="/nnunet_data/nnUNet_raw/Dataset999_20LongitudinalCT_dummy/labelsTr",
        predictions_dir="/nnunet_data/nnUNet_raw/Dataset999_20LongitudinalCT_dummy/predictions",
        output_dir="/nnunet_data/nnUNet_raw/Dataset999_20LongitudinalCT_dummy/patch_analysis",
        padding_voxels=40,
        target_spacing_mm=1.0,
    )
