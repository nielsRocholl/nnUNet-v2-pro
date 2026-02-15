# Step 4: Training Dataloader (Stochastic Local Sampler)

Prompt-aware stochastic patch sampling with four modes: pos, pos+spurious, pos+no-prompt, negative.

## Changes

**Modified:**
- [nnunetv2/utilities/roi_config.py](../nnunetv2/utilities/roi_config.py) — added `SamplingConfig`, `_load_sampling`, extended `RoiPromptConfig` and `load_config`
- [tests/fixtures/nnunet_pro_config.json](../tests/fixtures/nnunet_pro_config.json) — added `sampling.mode_probs`, `sampling.n_spur`, `sampling.n_neg`

**New:**
- [nnunetv2/training/dataloading/prompt_aware_data_loader.py](../nnunetv2/training/dataloading/prompt_aware_data_loader.py) — `nnUNetPromptAwareDataLoader`

**prompt_aware_data_loader:**
- Subclasses `nnUNetDataLoader`, overrides `determine_shapes` (+1 prompt channel), `generate_train_batch`
- Mode selection via `cfg.sampling.mode_probs`; fallback pos↔negative when case has no fg/bg
- Per mode: pos (centroids or fallback voxel), pos+spurious (+n_spur from B), pos+no-prompt (∅), negative (n_neg from Ω)
- Uses `encode_points_to_heatmap` from prompt_encoding; concatenates prompt to image before transforms

## Test

`pytest tests/test_step04_prompt_aware_dataloader.py`

**Visual output** (`test_step04_visual_output`) writes to `tests/outputs/step04/`:
- `patch_image_ch0.nii.gz` — first image channel
- `patch_prompt.nii.gz` — prompt heatmap (overlay on patch_image in CT viewer)
- `patch_label.nii.gz` — segmentation
- `README.txt` — overlay instructions

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 4
- Base: [nnunetv2/training/dataloading/data_loader.py](../nnunetv2/training/dataloading/data_loader.py)
- Step 3: [nnunetv2/utilities/prompt_encoding.py](../nnunetv2/utilities/prompt_encoding.py)
