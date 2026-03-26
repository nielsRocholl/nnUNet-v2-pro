# Precomputed lesion centroids (`*_centroids.json`)

Prompt-aware training samples patches using lesion centroids (and related geometry). For each preprocessed case, the dataloader can read **offline** centroid and bounding-box data instead of running connected-components analysis on every epoch.

## What gets written

For each case identifier `ID` next to `ID_seg.b2nd`, the pipeline writes **`ID_centroids.json`** in the same folder. The JSON contains:

- `centroids_zyx`: list of `[z, y, x]` integer centroids (one per lesion component)
- `bboxes_zyx`: list of `[zmin, zmax, ymin, ymax, xmin, xmax]` axis-aligned boxes (same order as centroids)

If a file is missing, the dataloader falls back to computing centroids from the segmentation at runtime (slower, same semantics when data match).

## Non-standalone (automatic with preprocess)

After each configuration’s preprocessing finishes, `nnUNetv2_plan_and_preprocess` calls `precompute_centroids_for_folder` on that configuration’s output directory. It uses the same **`nnUNetv2_plan_and_preprocess` worker count** (`-np` / `num_processes`) you passed for preprocessing, and respects **`--resume`**: existing `*_centroids.json` files are skipped when resume is enabled.

No extra CLI step is required for the usual workflow:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlannerResEncL -c 3d_fullres --np 8
```

## Standalone CLI

If you copied preprocessed data without going through this codebase’s preprocess, or you need to regenerate centroids after editing labels, use:

```bash
nnUNetv2_precompute_centroids -d DATASET_ID -plans_name nnUNetPlans -c 3d_fullres -np 8 [--resume]
```

| Argument | Description |
|----------|-------------|
| `-d` | Dataset ID (integer, e.g. `999`) |
| `-plans_name` | Plans JSON stem (default `nnUNetPlans`) |
| `-c` | Configuration name (e.g. `3d_fullres`, must match the folder under the dataset) |
| `-np` | Number of worker processes (default `8`) |
| `--resume` | Skip cases that already have `*_centroids.json` |

The tool resolves the folder as:

`$nnUNet_preprocessed/<DatasetName>/<data_identifier>/`

where `data_identifier` comes from the plans entry for `-c`.

## Requirements

- Blosc2 segmentations (`*_seg.b2nd`) must exist in the target folder (standard nnU-Net v2 pro preprocessing layout).
- `nnUNet_preprocessed` must point at the root that contains `DatasetXXX_.../`.
