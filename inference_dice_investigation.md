# Inference Dice Investigation Log

**Problem:** Model achieves Dice ~0.85 during validation, but inference (full mode, zero prompt) yields Dice ~0.001–0.01 on the same training data.

**Last updated:** 2026-03-02

---

## Root cause: training vs inference patch distribution

### Conclusion

The hypothesis matches the code:

1. **Training and validation Dice are patch-level** — They are computed on batches of patches (sum of tp/fp/fn over patches), not on full images.

2. **Patch distribution is heavily foreground-biased** — ~85% of patches are centered on foreground; only ~15% are random (often background).

3. **Inference uses full-volume sliding** — Every patch in the volume is processed; most patches are mostly background, which the model rarely saw during training/validation.

4. **Over-segmentation** — If the model predicts foreground in many background patches (because it was never trained to reject them), you get massive FP. On full-image Dice this destroys the score, even though patch-level Dice looked good.

The model is trained and validated on a foreground-heavy patch distribution, but evaluated at inference on a full-volume, background-heavy distribution. This mismatch explains the collapse from ~0.85 Dice in validation to near-zero Dice in inference.

---

## force_fg: original nnUNet vs prompt-aware

### What is force_fg?

`force_fg` is **original nnUNet code** (verified in `/root/nnunet/original_nnunet`). It is a parameter to `get_bbox` in the base `nnUNetDataLoader`:

- **`force_fg=False`**: Sample a random patch anywhere in the volume (can be mostly background).
- **`force_fg=True`**: Pick a random foreground voxel from `class_locations` and center the patch on it — the patch is guaranteed to contain foreground.

### Original nnUNet

- Uses `oversample_foreground_percent = 0.33`
- Per batch, the last ~33% of samples get `force_fg=True` (foreground)
- The rest (~67%) get `force_fg=False` (random)
- **Result: ~33% foreground patches, ~67% random patches**

### Prompt-aware nnUNet

- Uses `mode_probs = [0.5, 0.2, 0.15, 0.15]` (pos, pos+spurious, pos+no-prompt, negative)
- `force_fg = mode != MODE_NEG` — foreground for pos/pos+spur/pos+no-prompt, random for negative
- **Result: ~85% foreground patches, ~15% random patches**

### Effect on our model

| Aspect | Original nnUNet | Prompt-aware nnUNet |
|--------|----------------|---------------------|
| Foreground patches | ~33% | ~85% |
| Random/background patches | ~67% | ~15% |

The prompt-aware model is trained on a **much more foreground-heavy** distribution than original nnUNet. During inference it sees mostly background patches (full-volume sliding). This mismatch likely contributes to over-segmentation.

**Note:** `oversample_foreground_percent` is still passed to the prompt-aware dataloader but is never used — `force_fg` is determined entirely by `mode_probs` in `_get_bbox_and_mode`.

---

## Summary of Tests

| # | Test | Outcome | Notes |
|---|------|---------|-------|
| 1 | Step 3: Inspect prediction content | Done | Model predicts ~1.1M foreground voxels; GT has ~25k. Massive over-segmentation. Dice 0.0017. |
| 2 | Step 2: Compare preprocessing (raw vs saved) | Done | Raw and saved preprocessed data identical (max diff 0). **Preprocessing ruled out.** |
| 3 | Step 1: Inference from preprocessed data | Done | Dice still ~0.0017. **Data source ruled out** – same Dice from preprocessed as from raw. |
| 4 | Step 4: Export and metric verification | Done | ref_shape=pred_shape, spacing/origin match. **Export correct.** ref_foreground=12k, pred_foreground=1.1M. |
| 5 | Step 5: Normalization sanity check | Done | Raw -1024 to 3071, preprocessed -3.2 to 5.0. **Normalization OK.** |

---

## Theories Tested

| Theory | Status | Result |
|--------|--------|--------|
| ROI vs full sliding causes low Dice | Tested | Full mode (zero prompt) also gives ~0 Dice → **ruled out** |
| Centroid axis order (cc3d xyz vs zyx) | Tested | Reverted swap; `inside_mask=True` for all lesions → **ruled out** |
| Preprocessed vs on-the-fly preprocessing mismatch | Tested | Step 2: identical. Step 1: same Dice from preprocessed → **ruled out** |
| Export/coordinate space mismatch | Tested | Step 4: shapes, spacing, origin match → **ruled out** |
| Prediction content (all zeros?) | Tested | **Not all zeros** – 1.1M foreground voxels predicted; GT ~12k. Model over-segments. |
| Normalization mismatch | Tested | Step 5: raw→preprocessed range correct → **ruled out** |

---

## Findings

### Confirmed facts
- Validation uses zero prompt + full sliding; model sees no prompts during validation
- Training uses real prompts in ~70% of batches (MODE_POS, MODE_POS_SPUR)
- Dataset999_Merged includes Dataset013; our 5 test cases are in the merge
- Centroids are correct (inside_mask=True) after reverting axis swap
- **Root cause identified:** Training/validation use patch-level Dice with ~85% foreground patches; inference uses full-image Dice with mostly background patches. Prompt-aware model is more foreground-heavy than original nnUNet (85% vs 33%).

### Open questions
- Are our 5 test cases in the validation fold? Validation Dice is on the 20% val fold; if our 5 cases are in the 80% training fold, we are comparing different cases.

---

## Per-dataset analysis: over-segmentation diagnosis

**Test:** Ran `scripts/analyze_predictions.py` on full-mode and ROI-mode inference outputs (2 samples per dataset, 19 datasets).

**Result:** **26 over-segmentation, 0 under-segmentation, 12 other**

The model predicts far too many foreground voxels (n_pred >> n_ref). For small-lesion datasets, the model often predicts most of the organ/volume instead of the target.

| Pattern | Count | Interpretation |
|---------|-------|----------------|
| OVER: pred>>ref | 20 | n_pred 10–1000× n_ref |
| OVER: many FP | 6 | High FP, prediction much larger than reference |
| mixed/wrong loc | 12 | Includes good performers (LIDC, TCIA, LiTS) |
| UNDER | 0 | No under-segmentation |
| no GT | 2 | Dataset017: ref empty, model predicts ~20M voxels |

**Examples:**

| Dataset | n_ref | n_pred | Ratio | Interpretation |
|---------|-------|--------|-------|-----------------|
| Dataset013_Longitudinal_CT | 22k | 2.7M | 120× | Severe over-segmentation |
| Dataset012_LNDb | 254 | 4.7M | 18,000× | Extreme over-segmentation |
| Dataset015_MSD_Lung | 8k | 8.7M | 1,000× | Extreme over-segmentation |
| Dataset024_LIDC | 42M | 38M | ~1× | Good Dice (0.93) – organ-sized targets |
| Dataset025_TCIA | 66M | 52M | ~1× | Good Dice (0.88) – organ-sized targets |

**Conclusion:** The model tends to predict large foreground regions instead of focal lesions. Datasets with good Dice (LIDC, TCIA, LiTS) have organ-sized targets where over-segmentation still overlaps well with GT. Small-lesion datasets (Longitudinal CT, LNDb, MSD Lung, etc.) fail because the model predicts too much.

---

## Visual pattern: box vs blobs

### Observation

When inspecting full-volume predictions visually:

- **Box-like good region** — Lesions and nearby background are well segmented. There is a roughly box-shaped region around each lesion where the prediction matches GT.
- **Distant blobs** — Large “blobs” of false positives appear in regions far from any lesion.

### Patch-based explanation

- **Training**: ~85% of patches are centered on foreground (lesions). The model learns to segment lesions when the patch contains them.
- **Full-volume sliding**: Every patch in the volume is processed. Patches overlapping a lesion behave like training (good). Patches in distant background regions rarely saw foreground during training.
- **Result**: Patches overlapping lesions → good segmentation. Patches in distant background → model predicts foreground (FP blobs) because it was not trained to reject background in those regions.

The “box” corresponds to the union of sliding-window patches that overlap the lesion. The “blobs” correspond to distant background patches where the model over-segments.

---

## Raw Notes

(Add detailed observations, command outputs, and debug prints here as we investigate.)

--- Step 3: Inspect prediction content ---

Case: Longitudinal_CT_006f52e910_BL_img_BL_img_00
  Preprocessed data: shape=torch.Size([1, 490, 508, 508]), min=-3.2049, max=5.0024, mean=-2.0967
  Logits: shape=(2, 490, 508, 508), min=-8.0625, max=8.1016
  Logits > 0 count: 126681722
  Segmentation unique: {0: 63087363, 1: 1137917}
  Foreground voxels: 1137917
  Dice: 0.0017

Case: Longitudinal_CT_006f52e910_FU_img_FU_img_00
  Preprocessed data: shape=torch.Size([1, 482, 520, 520]), min=-3.2049, max=5.0024, mean=-2.1377

--- Step 2: Compare preprocessing (raw vs saved) ---
Case: Longitudinal_CT_006f52e910_BL_img_BL_img_00
  Raw: shape=(1, 490, 508, 508), min=-3.2049, max=5.0024, mean=-2.0968
  Saved: shape=(1, 490, 508, 508), min=-3.2049, max=5.0024, mean=-2.0968
  Shape match: True
  Max diff: 0.000000, mean diff: 0.000000
  Match (max_diff<1e-4): True

--- Step 1: Inference from preprocessed data ---
  Logits: shape=(2, 482, 520, 520), min=-7.5742, max=7.5781
  Logits > 0 count: 130552783
  Segmentation unique: {0: 62092049, 1: 1084655}
  Foreground voxels: 1084655
  Dice: 0.0000

--- Step 2: Compare preprocessing (raw vs saved) ---
  Raw vs saved (b2nd): shape_match=True, max_diff=0.0
  caseid: Longitudinal_CT_006f52e910_BL_img_BL_img_00
  raw_shape: [1, 490, 508, 508]
  raw_min: -3.2048840522766113
  raw_max: 5.002372741699219
  raw_mean: -2.0967776775360107
  preprocessed_found: True
  match: True
  saved_shape: [1, 490, 508, 508]
  saved_min: -3.2048840522766113
  saved_max: 5.002372741699219
  saved_mean: -2.0967776775360107
  shape_match: True
  max_diff: 0.0
  mean_diff: 0.0
  Longitudinal_CT_006f52e910_BL_img_BL_img_00: Dice=0.0017

--- Step 4: Export and metric verification ---
  caseid: Longitudinal_CT_006f52e910_BL_img_BL_img_00
  ref_shape: [1, 245, 512, 512], pred_shape: [1, 245, 512, 512], shape_match: True
  props_has_sitk_stuff: True
  sitk_spacing: [0.775390625, 0.775390625, 3.0], sitk_origin: [-196.11, -340.61, 1355.5]
  ref_spacing/origin match pred_spacing/origin: True
  ref_foreground: 12649, pred_foreground: 1137917

--- Step 5: Normalization sanity check ---
  raw_min: -1024.0, raw_max: 3071.0, raw_mean: -635.68
  preprocessed_min: -3.20, preprocessed_max: 5.00, preprocessed_mean: -2.10
  Longitudinal_CT_013d407166_BL_img_BL_img_00: Dice=0.0125
  Longitudinal_CT_02522a2b27_BL_img_BL_img_00: Dice=0.0014

Results saved to /home/nielsrocholl/diagnose_output/diagnose_results.json
