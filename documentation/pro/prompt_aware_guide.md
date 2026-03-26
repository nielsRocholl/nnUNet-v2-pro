# ROI-Prompted Segmentation Guide

This guide explains the prompt-aware extension added in nnU-Net v2 Pro. It lets you train and run inference with optional point prompts (e.g. lesion centroids) to improve segmentation, especially for lesions.

## Overview

The extension adds:

1. **Prompt channels** — Two extra input channels (pos + neg) for point heatmaps (concatenated with the image)
2. **Prompt-aware training** — Patches sampled with different prompt conditions (clean, noisy, missing, negative)
3. **ROI-only inference** — Sliding windows only over a dilated region around the prompt, never full-volume

Standard nnU-Net workflows (preprocessing, default training, standard inference) are unchanged. The prompt-aware path is opt-in via `nnUNetTrainerPromptAware` and `nnUNetv2_predict_roi`.

---

## Configuration: nnunet_pro_config.json

All tunable parameters live in a single JSON config. Use `tests/fixtures/nnunet_pro_config.json` as a template.

### Structure

```json
{
  "prompt": {
    "point_radius_vox": 2,
    "encoding": "edt",
    "validation_use_prompt": true,
    "prompt_intensity_scale": 0.5
  },
  "sampling": {
    "mode_probs": [0.35, 0.15, 0.15, 0.35],
    "n_spur": [1, 2],
    "n_neg": [1, 3],
    "large_lesion": {"K": 2, "K_min": 1, "K_max": 4, "max_extra": 0},
    "propagated": {"sigma_per_axis": [2.75, 5.19, 5.40], "max_vox": 34.0}
  },
  "inference": {
    "tile_step_size": 0.75,
    "disable_tta_default": false
  }
}
```

### Sections

| Section | Keys | Purpose |
|---------|------|---------|
| `prompt` | `point_radius_vox`, `encoding` (`binary` or `edt`), `prompt_intensity_scale` (0–1, default 1.0) | How points are encoded into a heatmap; intensity scale reduces over-reliance on prompts |
| `sampling` | `mode_probs`, `n_spur`, `n_neg`, `large_lesion`, `propagated` | Training patch sampling (see below) |
| `inference` | `tile_step_size`, `disable_tta_default` | Sliding window step, TTA default |

`sampling.propagated` (optional): `sigma_per_axis` `[σ_z, σ_y, σ_x]`, `max_vox` — offset distribution for simulated propagated prompts (defaults from longitudinal COG analysis).

### Large-lesion oversampling (`sampling.large_lesion.max_extra`)

Keep **`max_extra` at `0`** for real training. The code can append extra positive patches per step when `max_extra > 0`, which changes the **effective batch size** from step to step. That breaks trainers and models that assume a **fixed** batch shape (including fixed validation batches). The bundled default and the main test fixture use `max_extra: 0`. A separate test-only config enables extras for unit tests only.

### Offline centroids

After preprocessing, each case should have **`{case}_centroids.json`** next to `{case}_seg.b2nd` (written automatically by `nnUNetv2_plan_and_preprocess`, or via the standalone `nnUNetv2_precompute_centroids` CLI). See [precompute_centroids.md](precompute_centroids.md).

---

## Training

### 1. Standard preprocessing

Run nnU-Net preprocessing as usual. ResEnc L is recommended. Centroid JSON files are written automatically at the end of each configuration stage:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlannerResEncL -c 3d_fullres
```

### 2. Train with nnUNetTrainerPromptAware

Use `-p nnUNetResEncUNetLPlans` when preprocessing used ResEnc L. `--config` is optional; when omitted, the bundled default (EDT encoding) is used:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerPromptAware -p nnUNetResEncUNetLPlans
```

With a custom config:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerPromptAware -p nnUNetResEncUNetLPlans \
  --config path/to/nnunet_pro_config.json
```

The config is copied into the model folder as `nnunet_pro_config.json`, so inference can use it without `--config`.

### Stretched-tail poly learning rate (optional)

`nnUNetTrainerPromptAware` uses the default nnU-Net optimizer path (`nnUNetTrainer.configure_optimizers`). You can switch the LR schedule with **PRO** CLI flags:

- `--lr-schedule poly` (default) — standard polynomial decay over `num_epochs`.
- `--lr-schedule stretched_tail_poly` — same poly curve as if training for `--lr-stretched-ref` steps until epoch `--lr-stretched-k`, then a linear segment to the poly LR at `ref - 1`, stretched over the remaining epochs. Defaults: `--lr-stretched-k 750`, `--lr-stretched-ref 1000`, `--lr-stretched-exp 0.9`.

Example:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerPromptAware -p nnUNetResEncUNetLPlans \
  --lr-schedule stretched_tail_poly --epochs 2000
```

Trainers that **override** `configure_optimizers` (e.g. Adam, cosine, Primus/warmup) are unchanged by these flags. When continuing with `--c`, use the same `--lr-schedule` and stretched-tail arguments as the original run.

### 3. What the trainer does

- **Network**: Same architecture as nnU-Net, but with `num_input_channels + 2` (image + pos prompt + neg prompt).
- **Training patches**: Sampled with four modes:
  - **pos**: Patch with lesion, pos prompt = simulated propagated (centroid + random offset from propagation error distribution), neg prompt = zeros.
  - **pos+spurious**: Same as pos, plus `n_spur` spurious points in background.
  - **pos+no-prompt**: Patch with lesion, both prompt channels zeros.
  - **negative**: Patch without lesion, neg prompt = `n_neg` background points (from non-lesion voxels), pos prompt = zeros.

`mode_probs` controls the probability of each mode. Default [0.35, 0.15, 0.15, 0.35] gives 35% background patches to reduce over-segmentation (see inference_dice_investigation.md). The propagated offset simulates registration errors (e.g. baseline→follow-up COG propagation) for longitudinal inference. See `sampling.propagated` config.

- **Patch sampling**: Foreground patches use overlap sampling — the lesion may lie anywhere in the patch, not necessarily centered.

- **Large lesions**: The config still contains `large_lesion` (K, bbox logic). **Do not enable extra-patch oversampling** (`max_extra > 0`); see the configuration section above.
- **Validation**: Uses the same patch distribution (mode_probs) and prompts as training when `validation_use_prompt=true`. Per-mode Dice (pos, pos_spur, pos_no_prompt, neg, global) logged to WandB/logger.
- **Loss**: Standard nnU-Net Dice + cross-entropy.

---

## Inference

### CLI: nnUNetv2_predict_roi

```bash
nnUNetv2_predict_roi -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER \
  --points_json points.json [--config CONFIG] [--points_space voxel|world]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `-i` | Yes | Input folder or file (same channel layout as training) |
| `-o` | Yes | Output folder |
| `-m` | Yes | Model folder (trained with nnUNetTrainerPromptAware) |
| `--points_json` | Yes | Path to JSON with `points` and `points_space` |
| `--config` | No | Config path; default: `{model_folder}/nnunet_pro_config.json` |
| `--points_space` | No | Override `points_space` from JSON (`voxel` or `world`) |
| `--disable_tta` | No | Disable test-time augmentation |
| `--labels_folder` | No | Folder with ground truth for per-case DICE and running average |
| `-f` | No | Folds (default: 0) |
| `-chk` | No | Checkpoint name (default: checkpoint_final.pth) |
| `-device` | No | `cuda`, `cpu`, or `mps` |
| `--roi_mode` | No | Override `inference.roi_inference_mode`: `dilated_sliding` (default) or `per_point_patch` |

### ROI inference modes (`inference` config)

| Key | Default | Description |
|-----|---------|-------------|
| `roi_inference_mode` | `dilated_sliding` | `dilated_sliding`: one full-volume 2-channel prompt, sliding windows only over dilated bbox (existing behavior). `per_point_patch`: one local 2-channel prompt per forward, patch centered on each point, optional border expansion. |
| `max_patch_expansion_visits` | `64` | Max number of patch forwards per fold (seeds + expansions). When reached, expansion stops; with `--verbose`, a one-line notice is printed. |
| `max_patch_expansion_depth` | omitted | Optional max BFS depth from each seed; omit for no depth limit. |

**`per_point_patch` behavior (summary)**

1. Image is padded with `pad_nd_image` (image channels only). All patch indices use this **padded** grid; logits are cropped back before export (same world space as standard ROI).
2. Each seed point gets a `patch_size` window centered on the point (clamped at borders). Duplicate identical `(window, point)` pairs are skipped.
3. Empty `points` list: one centered patch with **zero** prompt channels (same spirit as empty prompt in dilated mode).
4. Each forward uses `encode_points_to_heatmap_pair` on **patch-local** coordinates (seed inside the patch, else patch center).
5. Predictions are merged with the same Gaussian weighting as nnU-Net sliding windows. Voxels never covered get **background-preferring** logits (no NaNs).
6. If foreground touches a **face** of the current patch (per `label_manager`), extra windows are queued, shifted by `patch_size//2` along that axis (outward), clamped. TTA matches existing ROI (`_internal_maybe_mirror_and_predict`).

### points.json format

```json
{
  "points": [[z, y, x], [z, y, x], ...],
  "points_space": "voxel",
  "points_format": "zyx_voxel"
}
```

- **voxel**: Coordinates in preprocessed voxel space `(z, y, x)`.
- **world**: Coordinates in world space `(x, y, z)` in mm; converted internally.
- **points_format** (optional): `zyx_voxel`, `xyz_voxel`, `xyz_world`, or `zyx_world`. Default: `zyx_voxel` for voxel, `xyz_world` for world. See [Coordinate validation](#coordinate-validation) below.

### Example

```bash
# Voxel: preprocessed (z,y,x)
echo '{"points": [[60, 125, 125]], "points_space": "voxel"}' > points.json

nnUNetv2_predict_roi -i $nnUNet_raw/Dataset010/imagesTr -o ./predictions_roi \
  -m $nnUNet_results/Dataset010/nnUNetTrainerPromptAware__nnUNetResEncUNetLPlans__3d_fullres \
  -f 0 --points_json points.json
```

If the model was trained with prompt-aware, the config is in the model folder and `--config` can be omitted.

### How inference works (`dilated_sliding`, default)

1. **Preprocessing**: Same as standard nnU-Net (resampling, normalization).
2. **Prompt heatmap**: All points are encoded into **one** full-volume 2-channel heatmap (pos + zeros for neg; radius `point_radius_vox`, encoding from config).
3. **Dilated bbox**: Bbox around prompt extent + `patch_size/2` per axis, clamped to image.
4. **Sliding windows**: Only windows overlapping this bbox are used.
5. **Single patch**: If the dilated bbox is smaller than the patch, a single centered patch is used.
6. **No full-volume sliding**: ROI mode never runs full-volume sliding.

For **`per_point_patch`**, see the table above (local prompts, queue + border expansion, Gaussian merge, safe background fill outside tiled regions).

---

## Inference display and per-sample DICE

Both **vanilla** (`nnUNetv2_predict`) and **ROI** (`nnUNetv2_predict_roi`) inference use Rich-formatted output (Panel, Progress bar, summary table) consistent with plan/preprocess/train.

### Per-sample DICE and running average

When ground truth labels are available, pass `--labels_folder` to print per-case DICE and a running mean:

- **ROI**: `nnUNetv2_predict_roi ... --labels_folder $nnUNet_raw/Dataset010/labelsTr`
- **Vanilla**: `nnUNetv2_predict -i ... -o ... -m ... --labels_folder $nnUNet_raw/Dataset010/labelsTr -npp 0 -nps 0`

Vanilla requires `-npp 0 -nps 0` (sequential mode) for DICE; multiprocessing mode does not support it. Labels must match input case IDs (e.g. `case_001.nii.gz` in labels folder for `case_001_0000.nii.gz` in images).

The summary table at the end includes **Mean Dice** when `--labels_folder` was provided and labels were found.

---

## Coordinate conventions

- **World/ITK**: `(x, y, z)` in mm.
- **Voxel/tensor**: `(z, y, x)` (NumPy/PyTorch order).

If `points_space` is `world`, conversion to preprocessed voxel space uses the image's spacing and affine.

---

## Coordinate validation

A common source of errors is mixing up **physical vs voxel** coordinates and **axis ordering** (x,y,z vs z,y,x). ROI inference validates and converts coordinates before use.

### Supported formats

| Format | Space | Order | Use case |
|--------|-------|-------|----------|
| `zyx_voxel` (default) | voxel | z,y,x | nnUNet preprocessed |
| `xyz_voxel` | voxel | x,y,z | ITK-SNAP, 3D Slicer voxel export |
| `xyz_world` (default for world) | world | x,y,z | ITK physical mm |
| `zyx_world` | world | z,y,x | Pipelines exporting world in array order |

Specify `points_format` in your JSON when your tool exports a different order:

```json
{"points": [[125, 125, 60]], "points_space": "voxel", "points_format": "xyz_voxel"}
```

### Common mistakes

- **Viewer exports (x,y,z) voxel** → use `points_format: "xyz_voxel"`.
- **Passing mm as voxel** → use `points_space: "world"` and `points_format: "xyz_world"` (or `zyx_world` if your source uses array order).
- **Wrong format** → validation raises `ValueError` with a clear message.

---

## Inference parameters

| Parameter | Location | Effect |
|-----------|----------|--------|
| `tile_step_size` | `inference.tile_step_size` | Sliding window step size (0.5 = 50% overlap, 1.0 = no overlap). Smaller = more patches, slower. |
| `disable_tta_default` | `inference.disable_tta_default` | Default TTA (mirroring) on/off. CLI `--disable_tta` overrides. |

---

## When to use

- **Lesion segmentation** with point prompts (e.g. from a detector or user clicks).
- **Semi-automatic workflows** where prompts are optional.
- **ROI-focused inference** when you only care about regions around prompts.

---

## Default nnU-Net unchanged

- Preprocessing: `nnUNetv2_plan_and_preprocess` — unchanged.
- Training: `nnUNetv2_train` with default trainer — unchanged.
- Inference: `nnUNetv2_predict` — unchanged.

The prompt-aware path is separate and only used when you choose `nnUNetTrainerPromptAware` and `nnUNetv2_predict_roi`.
