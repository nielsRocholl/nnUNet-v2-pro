# Step 10: autoPET IV Prompt Strategy

Smaller radius (2), negative channel, force_fg overlap (no centering bias), validation alignment, per-mode Dice logging.

## Changes

**Config:**
- [nnunetv2/utilities/nnunet_pro_config.json](../nnunetv2/utilities/nnunet_pro_config.json), [tests/fixtures/nnunet_pro_config.json](../tests/fixtures/nnunet_pro_config.json)
- `point_radius_vox`: 5 → 2
- `mode_probs`: [0.5, 0.2, 0.15, 0.15] → [0.35, 0.15, 0.15, 0.35] (35% background patches to reduce over-segmentation)
- `prompt_intensity_scale`: 0.5 — reduces prompt intensity to avoid over-reliance on prompts (autoPET-interactive does not use this)

**Architecture:**
- [nnUNetTrainerPromptChannel.py](../nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerPromptChannel.py) — `PROMPT_CHANNELS`: 1 → 2

**Encoding:**
- [prompt_encoding.py](../nnunetv2/utilities/prompt_encoding.py) — `encode_points_to_heatmap_pair(points_pos, points_neg, shape, ...)` for 2-channel (pos + neg)

**Dataloader:**
- [prompt_aware_data_loader.py](../nnunetv2/training/dataloading/prompt_aware_data_loader.py)
- Override `get_bbox`: foreground voxel anywhere in patch (overlap), not centered
- Two-channel encoding: MODE_NEG populates neg channel; pos modes use zeros for neg
- Add `mode` to batch for per-mode Dice logging

**Trainer:**
- [nnUNetTrainerPromptAware.py](../nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerPromptAware.py)
- `_prepare_validation_data`: append 2 zero channels
- Override `validation_step`, `on_validation_epoch_end`: per-mode Dice (pos, pos_spur, pos_no_prompt, neg, global)

**Inference:**
- [roi_predictor.py](../nnunetv2/inference/roi_predictor.py) — pos heatmap + zeros neg, concat 2 channels

## Test

`pytest tests/test_step10_autopet_prompt_strategy.py`

## References

- autoPET IV paper, autoPET-interactive codebase
- [inference_dice_investigation.md](../inference_dice_investigation.md) — background over-segmentation root cause
