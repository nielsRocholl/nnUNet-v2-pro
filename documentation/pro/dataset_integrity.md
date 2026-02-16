# Dataset Integrity Verification

During plan and preprocess, nnU-Net can verify that images and labels are geometrically consistent (spacing, origin, direction). This helps catch misaligned or corrupted cases before training.

## Flags

Use with `nnUNetv2_plan_and_preprocess` or `nnUNetv2_extract_fingerprint`:

- **`--verify_dataset_integrity`** — Run integrity checks. If any case fails, the pipeline stops.
- **`--reject_failing_cases`** — Run integrity checks and, for failing cases, move them to `imagesTr_rejected` / `labelsTr_rejected` and continue with the rest.

## Relaxed Geometry Tolerance

Geometry checks use relaxed tolerances (`rtol=1e-3`, `atol=1e-4`) so small floating‑point differences (e.g. `0.8` vs `0.80004883`) do not cause failures. This avoids false positives from typical DICOM/NIfTI conversion noise.

Larger mismatches (e.g. spacing `0.8` vs `0.9`) still fail.

## Usage

### Recommended: verify once per dataset

```bash
nnUNetv2_plan_and_preprocess -d 10 -pl nnUNetPlannerResEncL -c 3d_fullres --verify_dataset_integrity
```

### Reject failing cases and continue

When some cases have real geometry mismatches but you want to proceed with the rest:

```bash
nnUNetv2_plan_and_preprocess -d 10 -pl nnUNetPlannerResEncL -c 3d_fullres \
  --verify_dataset_integrity --reject_failing_cases
```

Failing cases are moved to `imagesTr_rejected` and `labelsTr_rejected` in the dataset folder, and `dataset.json` is updated accordingly.

### With merged datasets

```bash
nnUNetv2_plan_and_preprocess -d 10 11 12 --merge -o 999 -pl nnUNetPlannerResEncL -c 3d_fullres \
  --verify_dataset_integrity --reject_failing_cases
```

## What is checked

- **Spacing** — Image and label must match within tolerance
- **Origin** — For SimpleITK-based readers
- **Direction** — For SimpleITK-based readers
- **Affine** — For Nibabel-based readers
- **Shape** — Image and label must have the same spatial dimensions
- **NaN** — No NaN values allowed
- **Labels** — Only expected label IDs (from `dataset.json`)
