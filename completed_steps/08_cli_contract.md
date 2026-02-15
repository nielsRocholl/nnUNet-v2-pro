# Step 8: CLI Contract

Expose ROI mode with explicit point-space semantics. Wire training CLI for nnUNetTrainerPromptAware.

## Changes

**New:**
- [nnunetv2/inference/roi_predict_entrypoint.py](../nnunetv2/inference/roi_predict_entrypoint.py) — ROI-mode CLI
- [tests/test_step08_cli_roi.py](../tests/test_step08_cli_roi.py) — unit + integration tests
- [tests/test_e2e_train_and_predict_roi.py](../tests/test_e2e_train_and_predict_roi.py) — full E2E (train + predict)

**Modified:**
- [nnunetv2/run/run_training.py](../nnunetv2/run/run_training.py) — `--config`, `--epochs`, pass config_path to trainer
- [pyproject.toml](../pyproject.toml) — `nnUNetv2_predict_roi` script
- [tests/conftest.py](../tests/conftest.py) — e2e, slow markers

**roi_predict_entrypoint:**
- `-i`, `-o`, `-m`, `--config` (required), `--points_json` (required), `--points_space` (override), `--disable_tta`
- Load config first; CLI overrides apply second
- Preprocess via nnUNet iterator; parse points; world→voxel if needed; `predict_logits_roi_mode`; export

**run_training:**
- `--config` required when `-tr nnUNetTrainerPromptAware`
- `--epochs` optional override for overfit testing

## Test

`pytest tests/test_step08_cli_roi.py`

E2E (slow): `pytest tests/test_e2e_train_and_predict_roi.py -m e2e`

**Visual output** (`test_step08_visual_output`) writes to `tests/outputs/step08/`:
- `prediction.nii.gz`, `points_used.json`, `README.txt`

## CLI Commands (Manual E2E)

```bash
export nnUNet_raw="/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets/nnUNet_raw"
export nnUNet_preprocessed="/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets/nnUNet_preprocessed"
export nnUNet_results="/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/dummy datasets/nnUNet_results"
```

**Train (overfit on 1 sample):**
```bash
nnUNetv2_train Dataset010 3d_fullres all -tr nnUNetTrainerPromptAware --config tests/fixtures/nnunet_pro_config.json --epochs 5 --disable_checkpointing -device cuda
```

**ROI predict:**
```bash
# Create points.json (voxel, preprocessed z,y,x):
echo '{"points": [[60, 125, 125]], "points_space": "voxel"}' > points.json

nnUNetv2_predict_roi -i $nnUNet_raw/Dataset010/imagesTr -o $nnUNet_raw/Dataset010/predictions_roi \
  -m $nnUNet_results/Dataset010/nnUNetTrainerPromptAware__nnUNetPlans__3d_fullres \
  -f 0 --config tests/fixtures/nnunet_pro_config.json --points_json points.json
```

Points: use GT centroid from preprocessed seg, or center of volume. For world coords: `"points_space": "world"` with `[x_mm, y_mm, z_mm]`.

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 8
- LesionLocator: [temp/LesionLocator/lesionlocator/inference/lesionlocator_segment.py](../temp/LesionLocator/lesionlocator/inference/lesionlocator_segment.py)
