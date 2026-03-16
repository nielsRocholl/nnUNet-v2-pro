#!/usr/bin/env python3
"""Inspect nnUNet raw datasets for label structure and seg shapes (binary vs multi-class, channel counts)."""
import json
import os
import sys
from pathlib import Path

# Try nnUNet_raw from env, else common paths
NNUNET_RAW = os.environ.get(
    "nnUNet_raw",
    "/data/oncology/experiments/universal-lesion-segmentation/nnUNet_raw",
)
if not os.path.isdir(NNUNET_RAW):
    NNUNET_RAW = os.environ.get(
        "nnUNet_raw",
        "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/universal-lesion-segmentation/nnUNet_raw",
    )

# Source datasets from plan_and_preprocess_999.sh merge
MERGE_IDS = list(range(10, 28))


def _get_dataset_names() -> list[str]:
    names = []
    if not os.path.isdir(NNUNET_RAW):
        return names
    for d in sorted(os.listdir(NNUNET_RAW)):
        if any(d.startswith(f"Dataset{did:03d}_") for did in MERGE_IDS) and os.path.isdir(
            os.path.join(NNUNET_RAW, d)
        ):
            names.append(d)
    return names


def get_seg_shape(seg_path: str) -> tuple | None:
    """Return (num_channels, *spatial) as would be used by nnUNet. None on failure."""
    try:
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        if arr.ndim == 3:
            return (1,) + tuple(arr.shape)
        return tuple(arr.shape)
    except Exception:
        try:
            import nibabel as nib
            arr = nib.load(seg_path).get_fdata()
            return tuple(arr.shape)
        except Exception:
            return None


def load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_unique_labels(seg_path: str) -> set | None:
    """Read a segmentation file and return unique label values."""
    try:
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        return set(int(x) for x in arr.flat)
    except Exception:
        try:
            import nibabel as nib
            arr = nib.load(seg_path).get_fdata()
            return set(int(x) for x in arr.flat)
        except Exception:
            return None


def inspect_dataset(name: str) -> dict:
    base = os.path.join(NNUNET_RAW, name)
    ds_json_path = os.path.join(base, "dataset.json")
    labels_dir = os.path.join(base, "labelsTr")

    out = {"name": name, "path": base, "exists": os.path.isdir(base)}
    if not out["exists"]:
        return out

    ds_json = load_json(ds_json_path)
    if ds_json is None:
        out["dataset_json"] = "NOT FOUND"
        return out

    labels = ds_json.get("labels", {})
    out["labels"] = labels
    out["num_labels"] = len([k for k in labels if k not in ("background", "ignore")])
    out["has_regions"] = any(
        isinstance(v, (list, tuple)) and len(v) > 1 for v in labels.values()
    )
    out["binary"] = out["num_labels"] == 1 and not out["has_regions"]

    # Sample label files: unique values and shapes
    if os.path.isdir(labels_dir):
        files = [f for f in sorted(os.listdir(labels_dir)) if f.endswith((".nii.gz", ".nii"))][:5]
        out["sample_unique_labels"] = {}
        out["sample_seg_shapes"] = {}
        out["seg_channels"] = set()
        for f in files:
            p = os.path.join(labels_dir, f)
            uniq = get_unique_labels(p)
            shape = get_seg_shape(p)
            if uniq is not None:
                out["sample_unique_labels"][f] = sorted(uniq)
            else:
                out["sample_unique_labels"][f] = "(failed)"
            if shape is not None:
                out["sample_seg_shapes"][f] = shape
                out["seg_channels"].add(shape[0])
            else:
                out["sample_seg_shapes"][f] = "(failed)"
    return out


def main():
    print(f"nnUNet_raw: {NNUNET_RAW}")
    print(f"Exists: {os.path.isdir(NNUNET_RAW)}\n")
    if not os.path.isdir(NNUNET_RAW):
        print("ERROR: nnUNet_raw not found. Set nnUNet_raw env or edit script.")
        sys.exit(1)

    datasets = _get_dataset_names()
    if not datasets:
        datasets = ["Dataset023_LiTS", "Dataset024_LIDC"]

    all_channels = {}
    for name in datasets:
        r = inspect_dataset(name)
        print("=" * 60)
        print(f"Dataset: {r['name']}")
        print(f"Path: {r['path']}")
        print(f"Exists: {r['exists']}")
        if not r["exists"]:
            print()
            continue

        print(f"Labels: {r.get('labels', 'N/A')}")
        print(f"Num foreground labels: {r.get('num_labels', 'N/A')}")
        print(f"Has regions (multi-label per class): {r.get('has_regions', 'N/A')}")
        print(f"Binary (single foreground): {r.get('binary', 'N/A')}")

        shapes = r.get("sample_seg_shapes", {})
        if shapes:
            print("Sample seg shapes (channels, *spatial):")
            for f, s in shapes.items():
                print(f"  {f}: {s}")
            ch = r.get("seg_channels", set())
            if ch:
                all_channels[name] = ch
                print(f"  -> channel counts in sample: {sorted(ch)}")

        samples = r.get("sample_unique_labels", {})
        if samples:
            print("Sample unique label values:")
            for f, vals in list(samples.items())[:3]:
                print(f"  {f}: {vals}")
        print()

    if all_channels:
        print("=" * 60)
        print("SUMMARY: seg channel counts per dataset")
        inconsistent = [n for n, ch in all_channels.items() if len(ch) > 1 or (ch and min(ch) != 1)]
        if inconsistent:
            print(f"  INCONSISTENT (multiple channel counts): {inconsistent}")
        ch_per_ds = {n: sorted(ch)[0] if len(ch) == 1 else sorted(ch) for n, ch in all_channels.items()}
        for n, c in sorted(ch_per_ds.items()):
            print(f"  {n}: {c}")
        all_vals = set()
        for ch in all_channels.values():
            all_vals.update(ch)
        if len(all_vals) > 1:
            print(f"\n  WARNING: Different datasets have different channel counts: {sorted(all_vals)}")
            print("  This can cause the 10 vs 12 validation error. Re-preprocess with consistent labels.")

    # Inspect preprocessed Dataset999_Merged seg shapes
    prep = os.environ.get("nnUNet_preprocessed", os.path.join(os.path.dirname(NNUNET_RAW), "nnUNet_preprocessed"))
    prep_999 = os.path.join(prep, "Dataset999_Merged")
    config_dirs = []
    if os.path.isdir(prep_999):
        for d in os.listdir(prep_999):
            p = os.path.join(prep_999, d)
            if os.path.isdir(p) and ("3d_fullres" in d or "3d_lowres" in d):
                config_dirs.append((d, p))
    if config_dirs:
        print("=" * 60)
        print("PREPROCESSED Dataset999_Merged: seg shapes from .pkl")
        for cfg_name, cfg_path in config_dirs[:1]:
            pkl_files = [f for f in os.listdir(cfg_path) if f.endswith(".pkl") and not f.startswith(".")][:30]
            shapes = {}
            for f in pkl_files:
                try:
                    import pickle
                    with open(os.path.join(cfg_path, f), "rb") as fp:
                        props = pickle.load(fp)
                    if "seg_shape" in props:
                        shapes[f] = props["seg_shape"]
                    elif "shape_after_cropping" in props:
                        shapes[f] = ("from_crop", props.get("shape_after_cropping"))
                except Exception as e:
                    shapes[f] = f"(err: {e})"
            if shapes:
                ch_counts = {}
                for fn, s in shapes.items():
                    if isinstance(s, (list, tuple)) and len(s) >= 1:
                        ch = s[0]
                        ch_counts[ch] = ch_counts.get(ch, 0) + 1
                print(f"  Config: {cfg_name}")
                print(f"  Sample seg_shape[0] (channels) distribution: {dict(sorted(ch_counts.items()))}")
                if len(ch_counts) > 1:
                    print(f"  INCONSISTENT! Channel counts: {list(ch_counts.keys())}")
                    for fn, s in list(shapes.items())[:10]:
                        print(f"    {fn}: {s}")


if __name__ == "__main__":
    main()
