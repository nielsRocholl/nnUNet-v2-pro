# Step 8: CLI — single-patch prompt inference

**Entry:** [nnunetv2/inference/single_patch_predict_entrypoint.py](../nnunetv2/inference/single_patch_predict_entrypoint.py) → `nnUNetv2_predict_single_patch` in [pyproject.toml](../pyproject.toml).

**E2E (slow):** `pytest tests/test_e2e_train_and_predict_single_patch.py -m e2e`

## Manual E2E

```bash
export nnUNet_raw="..."
export nnUNet_preprocessed="..."
export nnUNet_results="..."
```

**Train (overfit on 1 sample):**
```bash
nnUNetv2_train Dataset010 3d_fullres all -tr nnUNetTrainerPromptAware --config tests/fixtures/nnunet_pro_config.json --epochs 5 --disable_checkpointing -device cuda
```

**Single-patch predict:**
```bash
echo '{"points": [[60, 125, 125]], "points_space": "voxel"}' > points.json

nnUNetv2_predict_single_patch -i $nnUNet_raw/Dataset010/imagesTr -o $nnUNet_raw/Dataset010/predictions_single_patch \
  -m $nnUNet_results/Dataset010/nnUNetTrainerPromptAware__nnUNetPlans__3d_fullres \
  -f 0 --config tests/fixtures/nnunet_pro_config.json --points_json points.json
```

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 8 (historical)
