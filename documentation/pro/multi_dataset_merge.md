# Multi-Dataset Merge

Merge multiple nnUNet datasets into one for joint fingerprint, planning, and preprocessing—without copying raw files.

## Usage

```bash
nnUNetv2_plan_and_preprocess -d 1 2 3 4 5 --merge -o 999 -pl nnUNetPlannerResEncL -c 3d_fullres
```

- `-d` — source dataset IDs to merge
- `--merge` — treat them as one dataset
- `-o` — output dataset ID (e.g. `999`) or full name (e.g. `Dataset999_Combined`)

Creates a virtual merged dataset in `nnUNet_raw/Dataset999_Merged/` (only `dataset.json`; paths point to originals), then runs fingerprint, planning, and preprocessing on it.

## Requirements

Source datasets must have identical:

- `channel_names`
- `file_ending`
- `labels`

## Step-by-step

You can also run the steps separately:

```bash
nnUNetv2_extract_fingerprint -d 1 2 3 --merge -o 999
nnUNetv2_plan_experiment -d 999
nnUNetv2_preprocess -d 999 -c 3d_fullres 3d_lowres
```

## Dataset Statistics (Pro)

When preprocessing a merged dataset, per-case statistics are collected automatically and saved as `case_stats_{config}.json` in the preprocessed folder. These support stratified batch sampling by lesion size and dataset source.

**Collected per case:**
- `fg_voxels.total` — total foreground voxels
- `fg_voxels.min_cc` — smallest connected component size (voxels)
- `fg_voxels.max_cc` — largest connected component size (voxels)
- `fg_voxels.mean_cc` — mean connected component size (voxels)
- `dataset` — source dataset name
- `size_bin` — background / tiny / small / medium / large (from `max_cc` thresholds)

**Output:** `nnUNet_preprocessed/Dataset999_Merged/case_stats_3d_fullres.json` (and per config)

The merged `dataset.json` includes `source_datasets` to map case IDs to their source. Stats collection runs for both `plan_and_preprocess` and standalone `preprocess` when the target is a merged dataset.

## Training

After preprocessing, train on the merged dataset as usual. Use `-p nnUNetResEncUNetLPlans` when ResEnc L was used for planning:

```bash
nnUNetv2_train 999 3d_fullres 0 -p nnUNetResEncUNetLPlans
```

## Stratified training

For balanced batches across datasets and lesion sizes, use `nnUNetTrainerPromptAwareStratified`. Requires `case_stats_{config}.json` (produced automatically when preprocessing a merged dataset).

Use `--config` when preprocessing to enable percentile-based size bins; use `--config` when training to control batch makeup via `stratified.dataset_weights` and `stratified.size_bin_weights`. See [stratified_sampling_guide.md](stratified_sampling_guide.md).

```bash
nnUNetv2_plan_and_preprocess -d 1 2 --merge -o 999 -c 3d_fullres --config path/to/nnunet_pro_config.json
nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerPromptAwareStratified -p nnUNetResEncUNetLPlans --config path/to/nnunet_pro_config.json
```
