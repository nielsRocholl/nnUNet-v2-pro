# Step 7: ROI-Only Inference Entrypoint

LesionLocator-style full-image inference: prompt-aware sliding windows over dilated bbox; never full-volume sliding.

## Changes

**New:**
- [nnunetv2/inference/roi_predictor.py](../nnunetv2/inference/roi_predictor.py) — ROI predictor, `get_prompt_aware_slicers`, `parse_points_json`

**Modified:**
- [nnunetv2/utilities/roi_config.py](../nnunetv2/utilities/roi_config.py) — added `InferenceConfig`; removed `RoiConfig` (radius_mm_base, margin_mm)
- [tests/fixtures/nnunet_pro_config.json](../tests/fixtures/nnunet_pro_config.json) — removed `roi` section; added `inference.tile_step_size`, `inference.disable_tta_default`

**roi_predictor:**
- `get_prompt_aware_slicers`: dilated bbox (prompt extent + `patch_size/2` per axis, clamped to image bounds). Single patch if dilated bbox < patch; else filter sliding windows to overlap dilated bbox. Empty prompt → 1 centered patch (never full-volume).
- `parse_points_json`: load points from JSON (voxel or world)
- `nnUNetROIPredictor`: subclass of `nnUNetPredictor`, overrides `predict_sliding_window_return_logits` to raise (never full-volume)
- `predict_logits_roi_mode`: LesionLocator-style — full image, single merged prompt for all points, one sliding-window pass over dilated bbox region

## Test

`pytest tests/test_step07_roi_inference.py`

**Unit tests:** `get_prompt_aware_slicers` (small/large/empty prompt, dilated bbox, dilation guarantees center), `parse_points_json` (voxel/world), `predict_sliding_window_return_logits` raises

**Integration:** `test_roi_predictor_end_to_end` — real preprocessed data, centroid point, full-image prediction, assert shape

**Visual output** (`test_step07_visual_output`) writes to `tests/outputs/step07/`:
- `full_image.nii.gz` — full preprocessed image
- `prompt_heatmap.nii.gz` — prompt heatmap (overlay on full_image)
- `full_prediction.nii.gz` — full-volume prediction
- `full_prediction_original_space*.nii.gz` — for CT viewer overlay
- `README.txt` — overlay instructions

## Usage

```python
from nnunetv2.inference.roi_predictor import nnUNetROIPredictor, parse_points_json
from nnunetv2.utilities.roi_config import load_config

cfg = load_config("/path/to/nnunet_pro_config.json")
pred = nnUNetROIPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True)
pred.initialize_from_trained_model_folder(model_folder, use_folds=(0,))

points_zyx, space = parse_points_json("points.json")
if space == "world":
    points_zyx = points_to_centers_zyx(points_zyx, "world", props, shape, spacing, ...)

logits = pred.predict_logits_roi_mode(data, points_zyx, props, cfg)
# Use export_prediction_from_logits for original-space export
```

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 7
- LesionLocator: [temp/LesionLocator/lesionlocator/inference/lesionlocator_segment.py](../temp/LesionLocator/lesionlocator/inference/lesionlocator_segment.py)
- MangoTree: `lesionlocator_prompt_aware_sliding_window_slicers`
