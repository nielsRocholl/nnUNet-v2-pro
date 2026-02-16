# ROI-Prompted Segmentation Guide

This guide explains the prompt-aware extension added in nnU-Net v2 Pro. It lets you train and run inference with optional point prompts (e.g. lesion centroids) to improve segmentation, especially for lesions.

## Overview

The extension adds:

1. **Prompt channel** — One extra input channel for a point heatmap (concatenated with the image)
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
    "point_radius_vox": 5,
    "encoding": "binary",
    "validation_use_prompt": true
  },
  "sampling": {
    "mode_probs": [0.5, 0.2, 0.15, 0.15],
    "n_spur": [1, 2],
    "n_neg": [1, 3],
    "large_lesion": {"K": 2, "K_min": 1, "K_max": 4, "max_extra": 3}
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
| `prompt` | `point_radius_vox`, `encoding` (`binary` or `edt`) | How points are encoded into a heatmap |
| `sampling` | `mode_probs`, `n_spur`, `n_neg`, `large_lesion` | Training patch sampling (see below) |
| `inference` | `tile_step_size`, `disable_tta_default` | Sliding window step, TTA default |

---

## Training

### 1. Standard preprocessing

Run nnU-Net preprocessing as usual. ResEnc L is recommended:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlannerResEncL -c 3d_fullres
```

### 2. Train with nnUNetTrainerPromptAware

You must provide a config file via `--config`. Use `-p nnUNetResEncUNetLPlans` when preprocessing used ResEnc L:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerPromptAware -p nnUNetResEncUNetLPlans \
  --config path/to/nnunet_pro_config.json
```

The config is copied into the model folder as `nnunet_pro_config.json`, so inference can use it without `--config`.

### 3. What the trainer does

- **Network**: Same architecture as nnU-Net, but with `num_input_channels + 1` (image + prompt).
- **Training patches**: Sampled with four modes:
  - **pos** — Patch with lesion; prompt = lesion centroids (or one random foreground voxel if none).
  - **pos+spurious** — Same as pos, plus `n_spur` spurious points in background.
  - **pos+no-prompt** — Patch with lesion; prompt channel all zeros.
  - **negative** — Patch without lesion; prompt = `n_neg` random points (wrong by construction).

`mode_probs` controls the probability of each mode. This teaches the model to use prompts when correct and ignore them when wrong or missing.

- **Large lesions**: Lesions larger than the patch get extra positive patches to reduce truncation bias.
- **Validation**: Uses a zero prompt channel (no real prompts).
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
| `-f` | No | Folds (default: 0) |
| `-chk` | No | Checkpoint name (default: checkpoint_final.pth) |
| `-device` | No | `cuda`, `cpu`, or `mps` |

### points.json format

```json
{
  "points": [[z, y, x], [z, y, x], ...],
  "points_space": "voxel"
}
```

- **voxel**: Coordinates in preprocessed voxel space `(z, y, x)`.
- **world**: Coordinates in world space `(x, y, z)` in mm; converted internally.

### Example

```bash
# Voxel: preprocessed (z,y,x)
echo '{"points": [[60, 125, 125]], "points_space": "voxel"}' > points.json

nnUNetv2_predict_roi -i $nnUNet_raw/Dataset010/imagesTr -o ./predictions_roi \
  -m $nnUNet_results/Dataset010/nnUNetTrainerPromptAware__nnUNetResEncUNetLPlans__3d_fullres \
  -f 0 --points_json points.json
```

If the model was trained with prompt-aware, the config is in the model folder and `--config` can be omitted.

### How inference works

1. **Preprocessing**: Same as standard nnU-Net (resampling, normalization).
2. **Prompt heatmap**: Points are encoded into a heatmap (radius `point_radius_vox`, encoding from config).
3. **Dilated bbox**: Bbox around prompt extent + `patch_size/2` per axis, clamped to image.
4. **Sliding windows**: Only windows overlapping this bbox are used.
5. **Single patch**: If the dilated bbox is smaller than the patch, a single centered patch is used.
6. **No full-volume sliding**: ROI mode never runs full-volume sliding.

---

## Coordinate conventions

- **World/ITK**: `(x, y, z)` in mm.
- **Voxel/tensor**: `(z, y, x)` (NumPy/PyTorch order).

If `points_space` is `world`, conversion to preprocessed voxel space uses the image's spacing and affine.

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
