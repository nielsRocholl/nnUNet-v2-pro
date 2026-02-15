# Step 6: Trainer Variant

Wire the prompt-aware dataloader into training while keeping nnU-Net Dice+CE loss.

## Changed to

**Previously:** Test asserted `trainer.roi_cfg.radius_mm_base == 50`. ROI config was never used by training (dataloader uses prompt + sampling only).

**Now:** Removed `radius_mm_base` assertion. Config has no ROI section. No behavioral change.

## Changes

**Modified:**
- [nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py](../nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) — added `**kwargs` to `__init__`, extended `my_init_kwargs` loop to accept kwargs for subclass params; added `_prepare_validation_data` hook for subclasses
- [nnunetv2/training/dataloading/prompt_aware_data_loader.py](../nnunetv2/training/dataloading/prompt_aware_data_loader.py) — added `force_zero_prompt` for validation (zero prompt channel, no large-lesion extras)

**New:**
- [nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerPromptAware.py](../nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerPromptAware.py) — trainer variant

**nnUNetTrainerPromptAware:**
- Inherits from `nnUNetTrainerPromptChannel` (network +1 prompt channel)
- Requires `config_path` in `__init__`; loads `RoiPromptConfig` via `load_config`
- Overrides `get_dataloaders`: train uses `nnUNetPromptAwareDataLoader` (prompt-aware), val uses same with `force_zero_prompt=True`
- Keeps standard nnU-Net Dice+CE loss (unchanged)
- Overrides `_prepare_validation_data`: appends zero prompt channel before validation prediction

**Validation and the zero prompt channel:**
- The model expects 2 input channels (image + prompt). Base validation loads only the image (1 channel).
- `_prepare_validation_data` appends a zero-filled tensor as the second channel so the network receives valid input.
- During validation we never use real prompts — we use an empty (zero) prompt. This matches the "pos+no-prompt" training mode.
- The model is trained to segment from image evidence alone when the prompt is empty.

## Test

`pytest tests/test_step06_trainer_variant.py`

**Visual output** (`test_step06_visual_output`) writes to `tests/outputs/step06/`:
- `patch_image.nii.gz`, `patch_prompt.nii.gz`, `patch_label.nii.gz` — train batch
- `val_patch_image.nii.gz`, `val_patch_prompt.nii.gz`, `val_patch_label.nii.gz` — val batch (zero prompt)
- `README.txt` — overlay instructions for CT viewer

## Usage

```python
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPromptAware import nnUNetTrainerPromptAware

trainer = nnUNetTrainerPromptAware(
    plans=plans,
    configuration="3d_fullres",
    fold=0,
    dataset_json=dataset_json,
    config_path="/path/to/nnunet_pro_config.json",
)
trainer.initialize()
dl_tr, dl_val = trainer.get_dataloaders()
```

Note: `run_training.py` does not yet pass `config_path`; use `--config` in Step 8 (CLI contract).

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 6
- Base: [nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py](../nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py)
- MangoTree: `nnunet_dice_ce_loss`
