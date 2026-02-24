# COG Propagation Error Analysis

Distance between `cog_propagated` and `cog_fu` in **voxels** (per lesion, per row).

## Summary

- **Lesions with both COGs**: 2605
- **Mean distance (voxels)**: 9.25
- **Median distance (voxels)**: 3.70
- **Std**: 15.51
- **Min / Max**: 0.10 / 203.89

## Percentiles (voxels)

| Percentile | Distance (vox) |
|------------|----------------|
| 25th | 1.42 |
| 50th | 3.70 |
| 75th | 11.05 |
| 90th | 23.06 |
| 95th | 33.98 |
| 99th | 79.73 |

## Per-axis (|dx|, |dy|, |dz|) in voxels

| Axis | Mean | Median | Std | Max |
|------|------|--------|-----|-----|
| x | 5.40 | 1.67 | 11.75 | 199.13 |
| y | 5.19 | 1.79 | 9.30 | 161.56 |
| z | 2.75 | 0.87 | 6.15 | 114.80 |

## By lesion_type

| Lesion type | n | Mean (vox) | Median (vox) | Std |
|-------------|---|------------|--------------|-----|
| Adrenals | 33 | 10.76 | 7.33 | 9.83 |
| CNS | 3 | 42.18 | 46.26 | 12.08 |
| Heart | 2 | 2.36 | 1.31 | 1.05 |
| Kidney | 6 | 5.07 | 4.12 | 2.39 |
| Liver | 355 | 13.68 | 7.93 | 18.95 |
| Lung | 801 | 5.12 | 1.68 | 7.71 |
| Lymph node | 740 | 9.80 | 3.32 | 19.44 |
| Others | 104 | 15.28 | 9.63 | 16.53 |
| Skeleton | 96 | 3.36 | 1.59 | 4.78 |
| Soft tissue / Skin | 404 | 11.61 | 6.64 | 14.33 |
| Spleen | 24 | 11.14 | 8.38 | 9.58 |
| unclear | 37 | 13.34 | 3.93 | 21.84 |

## Recommendations for prompt simulation

To simulate propagated prompts during training, use a random offset from the true centroid:

- **Offset magnitude**: Sample from ~N(μ=9.2, σ=15.5) voxels, or use empirical percentiles.
- **Per-axis**: Offsets are anisotropic (z typically smaller). Consider σ_x≈5.4, σ_y≈5.2, σ_z≈2.8 for anisotropic sampling.
- **Conservative bounds**: 95th percentile ≈ 34.0 voxels; max observed ≈ 203.9 voxels.

Config suggestion: add `propagated_offset_std_vox` (e.g. 15.5) or `propagated_offset_max_vox` (e.g. 34.0) for sampling.
