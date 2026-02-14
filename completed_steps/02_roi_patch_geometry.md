# Step 2: Coordinate Conversion

Convert points (voxel/world) to preprocessed `(z,y,x)` centers.

## Changed to

**Previously:** Config-driven cubic ROI (`radius_mm_base`, `margin_mm`) — `resolve_roi_radius_mm` → `build_roi_slices` → `crop_to_roi` for visualization.

**Now:** Patch-sized crop centered on centroid — `_center_crop_bbox(center, patch_size, shape)`. No ROI config; crop extent from `patch_size` in plans.

## Changes

**New files:**
- [nnunetv2/utilities/roi_config.py](../nnunetv2/utilities/roi_config.py) — config dataclass + `load_config(path)` (JSON)
- [nnunetv2/utilities/roi_geometry.py](../nnunetv2/utilities/roi_geometry.py) — coordinate conversion
- [tests/fixtures/nnunet_pro_config.json](../tests/fixtures/nnunet_pro_config.json) — test config

**roi_config:**
- `RoiPromptConfig`: prompt, sampling, inference (no ROI section)
- `load_config(path)`: fail fast on missing keys

**roi_geometry:**
- `points_to_centers_zyx`: voxel/world → `(z,y,x)` in preprocessed space

## Test

`pytest tests/test_step02_roi_geometry.py`

**Test case:** `nnUNet_raw/Dataset010/imagesTr/MSD_Liver_liver_1_0000.nii.gz` and `labelsTr/MSD_Liver_liver_1.nii.gz`

`test_step02_visual_output` writes NIfTIs to `tests/outputs/step02/`:
- `full_image.nii.gz` — full CT (overlay with `label.nii.gz` or `roi_mask.nii.gz`)
- `label.nii.gz` — full ground-truth segmentation
- `roi_crop.nii.gz` — patch-sized crop (overlay with `roi_crop_label.nii.gz`)
- `roi_crop_label.nii.gz` — segmentation cropped to ROI (matches `roi_crop` geometry)
- `roi_mask.nii.gz` — binary ROI region (full volume, use with `full_image`)

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 2
- LesionLocator: [temp/LesionLocator/lesionlocator/utilities/prompt_handling/prompt_handler.py](../temp/LesionLocator/lesionlocator/utilities/prompt_handling/prompt_handler.py)
