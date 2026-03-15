# Updating Plans (patch_size, batch_size) Without Re-preprocessing

When you preprocessed with a different GPU memory target than you want for training, you can edit the plans file directly. **Preprocessed data is patch-size agnostic** — the dataloader crops patches at training time.

## When to Use

- **patch_size**: Reduce to fit inference on smaller GPUs (e.g. 20GB)
- **batch_size**: Increase to maximize throughput on large GPUs (e.g. 140GB H200)

## Script (Recommended)

```bash
python scripts/update_plans_patch_batch.py \
  --preprocessed-dir $nnUNet_preprocessed/Dataset999_Merged \
  --plans-name nnUNetResEncUNetLPlans_h200 \
  --config 3d_fullres \
  --patch-size 96 128 128 \
  --batch-size 8
```

### Example: 20GB inference + 140GB H200 training

- **patch_size** `96 128 128` — fits ~20GB VRAM (ResEncUNet 7-stage)
- **batch_size** `8` — fits ~140GB H200 (tune down if OOM)

### Divisibility

Patch size must be divisible by the network's `shape_must_be_divisible_by`. For ResEncUNet 7-stage:

- D (first axis): multiple of **32**
- H, W: multiple of **64**

Valid examples: `96 128 128`, `96 192 192`, `128 192 192`.

## Manual Edit

Edit `nnUNet_preprocessed/Dataset999_Merged/nnUNetResEncUNetLPlans_h200.json`:

```json
"configurations": {
  "3d_fullres": {
    "patch_size": [96, 128, 128],
    "batch_size": 8,
    ...
  }
}
```

## Notes

- **data_identifier** stays the same — preprocessed folder is reused
- **architecture** (strides, features) is fixed — only patch_size and batch_size can change
- If you change patch_size, ensure divisibility or training will fail
