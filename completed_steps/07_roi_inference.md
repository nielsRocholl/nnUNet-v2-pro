# Step 7: Prompt-aware `nnUNetROIPredictor` (single-patch path)

`nnunetv2/inference/roi_predictor.py` — `nnUNetROIPredictor` blocks full-volume sliding; use `predict_logits_single_patch` and helpers (`parse_points_json`, border expansion, heatmap encoding).

Dilated-bbox / `predict_logits_roi_mode` and `get_prompt_aware_slicers` were **removed**; Pro CLI is `nnUNetv2_predict_single_patch` only.

**Tests:** `tests/test_single_patch_*.py`, `tests/test_roi_patch_mode.py` (helpers + `InferenceConfig`).

**Usage:**

```python
from nnunetv2.inference.roi_predictor import nnUNetROIPredictor, parse_points_json
from nnunetv2.utilities.roi_config import load_config

cfg = load_config("/path/to/nnunet_pro_config.json")
pred = nnUNetROIPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True)
pred.initialize_from_trained_model_folder(model_folder, use_folds=(0,))
# Build points_zyx in preprocessed space, then:
logits = pred.predict_logits_single_patch(data, points_zyx, cfg, encode_prompt=False)
```

## References

- Plan: [SEGMENTATION_DESIGN_UPDATED.md](../SEGMENTATION_DESIGN_UPDATED.md) § 7 (historical)
