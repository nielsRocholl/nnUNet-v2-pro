# Step 1: Network Input Channels

Adds 1 extra input channel for the prompt heatmap via early concatenation, following nnInteractive's pattern.

## Changes

**New file:** [nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerPromptChannel.py](../nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerPromptChannel.py)

- Trainer subclass that overrides `build_network_architecture` to pass `num_input_channels + PROMPT_CHANNELS` (1) to the base implementation.
- Reuses `nnUNetTrainer.build_network_architecture` → `get_network_from_plans`; no architecture changes.
- `PROMPT_CHANNELS = 1` is configurable for future nnInteractive alignment (7 channels).

## Test

[tests/test_step01_network_input_channels.py](../tests/test_step01_network_input_channels.py) — `pytest tests/test_step01_network_input_channels.py`

`test_step01_visual_output` writes NIfTIs to `tests/outputs/step01/` for inspection:
- `input_ch0_image.nii.gz` — synthetic CT-like image
- `input_ch1_prompt.nii.gz` — nnInteractive-style sphere prompt [0,1]
- `output_logits_fg_minus_bg.nii.gz` — logit difference
- `output_pred.nii.gz` — binary prediction

## References

- Plan: [SEGMENTATION_DESIGN.md](../SEGMENTATION_DESIGN.md) § 1) Network Input Channels
- nnInteractive: [temp/nnInteractive/nnInteractive/trainer/nnInteractiveTrainer.py](../temp/nnInteractive/nnInteractive/trainer/nnInteractiveTrainer.py) — `num_input_channels + 7`
- Base: [nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py](../nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) L316–348
