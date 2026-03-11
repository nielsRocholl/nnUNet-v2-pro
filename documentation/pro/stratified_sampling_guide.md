# Stratified Sampling Guide

Control batch makeup by dataset and lesion size for merged-dataset training.

## Statistics Overview

When preprocessing a **merged** dataset, per-case statistics are collected and saved as `case_stats_{config}.json`:

| Field | Meaning |
|-------|---------|
| `fg_voxels.total` | Total foreground voxels in the segmentation |
| `fg_voxels.min_cc` | Smallest connected component size (voxels) |
| `fg_voxels.max_cc` | Largest connected component size (voxels) |
| `fg_voxels.mean_cc` | Mean connected component size |
| `dataset` | Source dataset name (e.g. `Dataset001_Mini`) |
| `size_bin` | `background` / `tiny` / `small` / `medium` / `large` |

**Size bin** is derived from `max_cc` (largest lesion). Bins can be fixed or data-driven (percentile-based).

## Size Bins

### Fixed mode (default)

Thresholds in voxels: `[100, 2000, 20000]`

- `tiny`: 1–99
- `small`: 100–1999
- `medium`: 2000–19999
- `large`: ≥20000

### Percentile mode

Compute thresholds from the dataset distribution after trimming extremes. Reduces impact of outliers.

```json
"size_bins": {
  "mode": "percentile",
  "trim_percentile": 0.025,
  "percentiles": [0.25, 0.5, 0.75]
}
```

- `trim_percentile`: Drop this fraction from each tail before computing percentiles (e.g. 0.025 = 2.5% each side)
- `percentiles`: Three values defining four bins (quartiles by default)

Use `--config path/to/nnunet_pro_config.json` when preprocessing a merged dataset to apply `size_bins`.

## Stratified Sampler

Control the proportion of each dataset and size bin in each batch.

```json
"stratified": {
  "dataset_weights": {"Dataset001_Mini": 0.5, "Dataset002_Mini": 0.5},
  "size_bin_weights": {"tiny": 0.1, "small": 0.3, "medium": 0.4, "large": 0.2, "background": 0.0}
}
```

- `dataset_weights`: Target proportion per source dataset. Must sum to 1. Omit datasets for uniform.
- `size_bin_weights`: Target proportion per size bin. Must sum to 1. Set `background` to 0 to exclude.

Per-stratum weight = `dataset_weights[dataset] × size_bin_weights[size_bin]`. Batches are sampled proportionally.

### Examples

**Equal datasets, emphasize medium lesions:**
```json
"stratified": {
  "dataset_weights": {"Dataset001_Mini": 0.5, "Dataset002_Mini": 0.5},
  "size_bin_weights": {"tiny": 0.1, "small": 0.2, "medium": 0.5, "large": 0.2, "background": 0.0}
}
```

**Favor one dataset:**
```json
"stratified": {
  "dataset_weights": {"Dataset001_Mini": 0.7, "Dataset002_Mini": 0.3},
  "size_bin_weights": {"tiny": 0.25, "small": 0.25, "medium": 0.25, "large": 0.25, "background": 0.0}
}
```

**Uniform (no config):** Omit `stratified` for round-robin over strata.

## End-to-End

1. **Preprocess** merged dataset with config (for percentile size bins):
   ```bash
   nnUNetv2_plan_and_preprocess -d 1 2 --merge -o 999 -c 3d_fullres --config path/to/nnunet_pro_config.json
   ```

2. **Train** with stratified sampler:
   ```bash
   nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerPromptAwareStratified -p nnUNetResEncUNetLPlans --config path/to/nnunet_pro_config.json
   ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: case_stats required` | Run preprocessing on a merged dataset first. `case_stats_{config}.json` is created automatically. |
| `No strata after filtering` | Ensure `tr_keys` overlap with case IDs in `case_stats`. Check `source_datasets` in merged `dataset.json`. |
| Percentile bins fallback to fixed | Need ≥10 non-background cases. Add more data or use `mode: "fixed"`. |
