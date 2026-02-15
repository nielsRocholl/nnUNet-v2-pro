# Step 3: Prompt Extraction and Encoding

Convert lesion instances into one dense prompt heatmap aligned with each sampled patch.

## Changed to

**Previously:** Config-driven cubic ROI crop around centroid — `resolve_roi_radius_mm` → `build_roi_slices` → `crop_to_roi` for `roi_img`, `roi_seg` before `build_prompt_channel`.

**Now:** Patch-sized crop centered on centroid — `_center_crop_bbox(center, patch_size, shape)`. No ROI config.

## Changes

**Modified:**
- [nnunetv2/utilities/roi_config.py](../nnunetv2/utilities/roi_config.py) — added `PromptConfig`, `RoiPromptConfig`, prompt section in `load_config`
- [tests/fixtures/nnunet_pro_config.json](../tests/fixtures/nnunet_pro_config.json) — added `prompt.point_radius_vox`, `prompt.encoding`
- [pyproject.toml](../pyproject.toml) — added `connected-components-3d`

**New:**
- [nnunetv2/utilities/prompt_encoding.py](../nnunetv2/utilities/prompt_encoding.py) — prompt extraction and encoding

**prompt_encoding:**
- `extract_centroids_from_seg`: LesionLocator logic via cc3d, returns `(z,y,x)` per centroid
- `filter_centroids_in_patch`: keep centroids inside slices, convert to patch-local coords
- `encode_points_to_heatmap`: ball + optional EDT, merge with `torch.maximum`, range [0,1]
- `build_prompt_channel`: full pipeline, returns `(1, D, H, W)` float32

## Test

`pytest tests/test_step03_prompt_encoding.py`

**Visual output** (`test_step03_visual_output`) writes to `tests/outputs/step03/`:
- `roi_crop.nii.gz` — ROI image crop
- `roi_crop_label.nii.gz` — segmentation cropped to ROI
- `roi_crop_prompt.nii.gz` — prompt heatmap (overlay on roi_crop in CT viewer)
- `full_prompt.nii.gz` — prompt at full volume (overlay on full_image)

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 3
- LesionLocator: [temp/LesionLocator/lesionlocator/utilities/prompt_handling/prompt_handler.py](../temp/LesionLocator/lesionlocator/utilities/prompt_handling/prompt_handler.py)
- nnInteractive: [temp/nnInteractive/nnInteractive/interaction/point.py](../temp/nnInteractive/nnInteractive/interaction/point.py)
- MangoTree: `lesionlocator_prompt_extraction_from_segmentation`
