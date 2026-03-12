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

### Percentile mode (default for merged datasets)

**Default.** Compute thresholds from the dataset distribution after trimming extremes. Reduces impact of outliers. No config required.

- `trim_percentile`: 0.025 (drop 2.5% from each tail before computing percentiles)
- `percentiles`: [0.25, 0.5, 0.75] (quartiles → four bins: tiny, small, medium, large)

To override, add `size_bins` to your config and pass `--config`:

```json
"size_bins": {
  "mode": "percentile",
  "trim_percentile": 0.025,
  "percentiles": [0.25, 0.5, 0.75]
}
```

### Fixed mode

Explicit voxel thresholds: `[100, 2000, 20000]`

- `tiny`: 1–99
- `small`: 100–1999
- `medium`: 2000–19999
- `large`: ≥20000

Use `mode: "fixed"` in config when you want fixed thresholds instead of data-driven percentiles:

```json
"size_bins": {
  "mode": "fixed",
  "thresholds": [100, 2000, 20000]
}
```

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

1. **Preprocess** merged dataset (percentile size bins are used by default; no config needed):
   ```bash
   nnUNetv2_plan_and_preprocess -d 1 2 --merge -o 999 -c 3d_fullres
   ```

2. **Train** with stratified sampler (use `--config` to control batch makeup):
   ```bash
   nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerPromptAwareStratified -p nnUNetResEncUNetLPlans --config path/to/nnunet_pro_config.json
   ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: case_stats required` | Run preprocessing on a merged dataset first. `case_stats_{config}.json` is created automatically. |
| `No strata after filtering` | Ensure `tr_keys` overlap with case IDs in `case_stats`. Check `source_datasets` in merged `dataset.json`. |
| Percentile bins fallback to fixed | Need ≥10 non-background cases. Add more data or use `mode: "fixed"`. |
