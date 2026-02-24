# Step 9: Propagated Prompt Simulation

Replace the pos mode's perfect COG prompt with a simulated propagated prompt that samples offsets from the propagation error distribution observed in longitudinal CT (cog_propagated vs cog_fu). Training matches inference reality where prompts come from registration propagation, not ground-truth centroids.

## Changes

**New:**
- [nnunetv2/utilities/propagated_prompt_simulation.py](../nnunetv2/utilities/propagated_prompt_simulation.py) — `apply_propagation_offset`

**Modified:**
- [nnunetv2/utilities/roi_config.py](../nnunetv2/utilities/roi_config.py) — added `PropagatedConfig`, `_load_propagated`, extended `SamplingConfig`
- [nnunetv2/training/dataloading/prompt_aware_data_loader.py](../nnunetv2/training/dataloading/prompt_aware_data_loader.py) — pos and pos+spurious use perturbed centroids; same in large-lesion extras
- [nnunetv2/utilities/nnunet_pro_config.json](../nnunetv2/utilities/nnunet_pro_config.json) — added `sampling.propagated`
- [tests/fixtures/nnunet_pro_config.json](../tests/fixtures/nnunet_pro_config.json) — added `sampling.propagated`
- [documentation/pro/prompt_aware_guide.md](../documentation/pro/prompt_aware_guide.md) — updated pos description, documented `propagated` config

**Scripts:**
- [scripts/test_propagated_prompt_on_dataset013.py](../scripts/test_propagated_prompt_on_dataset013.py) — test on preprocessed Dataset013
- [scripts/test_propagated_prompt_raw_dataset013.py](../scripts/test_propagated_prompt_raw_dataset013.py) — test on raw labelsTr-test / imagesTr-test

**prompt_aware_data_loader:**
- pos and pos+spurious: for each centroid, apply `apply_propagation_offset` before encoding
- Offset: anisotropic Gaussian N(0, σ²) per axis, truncated to max_vox magnitude, clipped to patch bounds
- Default params from [scripts/cog_propagation_analysis_report.md](../scripts/cog_propagation_analysis_report.md): σ=(2.75, 5.19, 5.40), max_vox=34

## Test

`pytest tests/test_propagated_prompt_simulation.py`

**Dataset013 real-sample test:**
```bash
# Raw (labelsTr-test / imagesTr-test):
python scripts/test_propagated_prompt_raw_dataset013.py

# Preprocessed:
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
python scripts/test_propagated_prompt_on_dataset013.py
```

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 9
- COG analysis: [scripts/cog_propagation_analysis_report.md](../scripts/cog_propagation_analysis_report.md)
- Step 4: [nnunetv2/training/dataloading/prompt_aware_data_loader.py](../nnunetv2/training/dataloading/prompt_aware_data_loader.py)
