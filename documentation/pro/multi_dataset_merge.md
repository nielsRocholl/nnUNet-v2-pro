# Multi-Dataset Merge

Merge multiple nnUNet datasets into one for joint fingerprint, planning, and preprocessing—without copying raw files.

## Usage

```bash
nnUNetv2_plan_and_preprocess -d 1 2 3 4 5 --merge -o 999
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

## Training

After preprocessing, train on the merged dataset as usual:

```bash
nnUNetv2_train 999 3d_fullres 0
```
