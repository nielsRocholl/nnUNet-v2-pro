# Step 5: Large-Lesion Add-On (Sparse Extra Patches)

Add bounded extra samples for lesions larger than one patch to avoid truncation bias.

## Changes

**Modified:**
- [nnunetv2/utilities/roi_config.py](../nnunetv2/utilities/roi_config.py) — added `LargeLesionConfig`, `_load_large_lesion`, extended `SamplingConfig` and `load_config`
- [nnunetv2/training/dataloading/prompt_aware_data_loader.py](../nnunetv2/training/dataloading/prompt_aware_data_loader.py) — `_sample_large_lesion_extras`, integrate extras in `generate_train_batch`
- [tests/fixtures/nnunet_pro_config.json](../tests/fixtures/nnunet_pro_config.json) — added `sampling.large_lesion`

**New:**
- [nnunetv2/utilities/large_lesion_sampling.py](../nnunetv2/utilities/large_lesion_sampling.py) — bbox extraction, is_large, sample_centers

**large_lesion_sampling:**
- `get_lesion_bboxes_zyx`: cc3d.statistics → per-instance bboxes (zmin,zmax, ymin,ymax, xmin,xmax)
- `is_large_lesion`: ∃k: Δ_k > P_k
- `sample_extra_centers_for_large_lesion`: coarse grid stride ⌊P_k/2⌋, filter to mask, sample K from [K_min,K_max], cap by max_extra per case

**prompt_aware_data_loader:**
- After main batch: for each case, get bboxes, filter large lesions, sample centers, crop patches, build pos-mode prompt, append to batch
- Variable batch size: batch_size + n_extras
- Transforms loop uses `data_all.shape[0]` instead of `batch_size`

## Test

`pytest tests/test_step05_large_lesion_sampling.py`

**Unit tests:** config load, get_lesion_bboxes_zyx, is_large_lesion, sample_extra_centers

**Integration:** synthetic case with large lesion (60×220×220) vs patch (48×192×192); assert extras added

**Visual output** (`test_step05_visual_output`) writes to `tests/outputs/step05/`:
- `extra_patch_0_image.nii.gz`, `extra_patch_0_prompt.nii.gz`, `extra_patch_0_label.nii.gz`
- `large_lesion_mask.nii.gz` — full-volume mask for overlay
- `README.txt` — overlay instructions

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 5
- LesionLocator: [temp/LesionLocator/lesionlocator/modules/tracknet.py](../temp/LesionLocator/lesionlocator/modules/tracknet.py)
- MangoTree: `lesionlocator_patch_extraction_around_mask`
