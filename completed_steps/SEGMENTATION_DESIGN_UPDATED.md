# Segmentation Design

## General Explanation
Build an ROI-prompted 3D nnU-Net segmentation module. Prompts are optional weak priors. Training uses stochastic local patches (nnU-Net style) plus prompt-aware sampling. Inference is ROI-only and uses prompt-aware local sliding windows. No full-volume sliding fallback in ROI mode.

## Invariants
- ROI mode never runs full-volume sliding.
- Prompt channel is optional (`all-zero` is valid).
- Output is full-volume lesion mask (inference runs prompt-aware sliding over dilated bbox only).
- Prefer existing code in `temp` and MangoTree over new abstractions.

## Configuration Contract (Mandatory)
- Never hardcode tunable numeric parameters in training/inference code.
- Load all tunables from one config object (YAML/JSON -> dataclass), then pass it explicitly.
- CLI may override config keys, but resolved values must still come from the config object.
- If a key is missing: fail fast with a clear error (no silent fallback literals).
- Required tunable keys:
- `sampling.mode_probs`, `sampling.n_spur`, `sampling.n_neg`
- `sampling.large_lesion.K`, `sampling.large_lesion.K_min`, `sampling.large_lesion.K_max`, `sampling.large_lesion.max_extra`
- `sampling.propagated` (optional): `sigma_per_axis` [σ_z, σ_y, σ_x], `max_vox`. Defaults: [2.75, 5.19, 5.40], 34.0 — from longitudinal COG propagation error analysis (see § 9)
- `prompt.point_radius_vox`, `prompt.encoding` (`binary|edt`)
- `inference.disable_tta_default`, `inference.tile_step_size`
- Non-tunable by config: planned `patch_size` comes from nnU-Net plans.

## Axis Convention (Mandatory)
- World/ITK coordinates: `(x, y, z)`.
- NumPy/PyTorch tensor indexing: `(z, y, x)`.
- Internal patch/ROI bounds in this document are always `(z, y, x)`.
- Spacing vector used for tensor-space math is `(s_z, s_y, s_x)`.

Conversions:
$$
(x,y,z)_{\text{world/ITK}} \leftrightarrow (z,y,x)_{\text{tensor}}
$$
$$
[x_{\min},x_{\max},y_{\min},y_{\max},z_{\min},z_{\max}] \leftrightarrow [z_{\min},z_{\max},y_{\min},y_{\max},x_{\min},x_{\max}]
$$

## Ground-Truth Sources
- `temp/nnInteractive/nnInteractive/trainer/nnInteractiveTrainer.py`
- `temp/nnInteractive/nnInteractive/interaction/point.py`
- `temp/nnInteractive/nnInteractive/inference/inference_session.py`
- `temp/LesionLocator/lesionlocator/utilities/prompt_handling/prompt_handler.py`
- `temp/LesionLocator/lesionlocator/inference/lesionlocator_segment.py`
- `temp/LesionLocator/lesionlocator/modules/tracknet.py`
- MangoTree IDs: `nninteractive_early_prompting_channel_concatenation`, `coordinate_order_simpleitk_numpy_torch`, `lesionlocator_prompt_coordinate_transformation`, `lesionlocator_prompt_extraction_from_segmentation`, `lesionlocator_prompt_aware_sliding_window_slicers`, `totalsegmentator_multipart_inference`, `nnunet_dice_ce_loss`

If local code is ambiguous: call MangoTree `get_primitive_implementation`. If still ambiguous: open primitive GitHub link.

## Step-by-Step Implementation Plan

### 1) Network Input Channels (already implemented)
High-level:
- Inject prompt information early by adding one channel to the image input.

Do:
1. Keep network input as `C_img + 1` channels.
2. Last channel is prompt heatmap.

Check:
- `temp/nnInteractive/nnInteractive/trainer/nnInteractiveTrainer.py`
- MangoTree `nninteractive_early_prompting_channel_concatenation`

### 2) Coordinate Contract
High-level:
- Convert points to preprocessed voxel `(z,y,x)` centers.

Do:
1. Support `points_space ∈ {voxel, world}` (from JSON or CLI override).
2. If `world`, convert `(x,y,z)` to tensor `(z,y,x)` in preprocessed voxel space, then round/clamp.
3. Inference uses dilated bbox (prompt extent + `patch_size/2`), not ROI cropping.

Check:
- `nnunetv2/utilities/roi_geometry.py` — `points_to_centers_zyx`
- `temp/LesionLocator/lesionlocator/inference/lesionlocator_segment.py`

### 3) Prompt Extraction and Encoding
High-level:
- Convert lesion instances into one dense prompt heatmap aligned with each sampled patch.

Do:
1. Extract lesion centroids from GT via LesionLocator logic.
2. Keep centroids inside current patch only.
3. Encode all kept points into one heatmap channel (nnInteractive point logic).
4. Merge multiple points by `torch.maximum`.

Encoding constraints:
- Heatmap range `[0,1]`.
- Zero channel allowed.

Check:
- `temp/LesionLocator/lesionlocator/utilities/prompt_handling/prompt_handler.py`
- `temp/nnInteractive/nnInteractive/interaction/point.py`
- MangoTree `lesionlocator_prompt_extraction_from_segmentation`

### 4) Training Dataloader (stochastic local sampler)
High-level:
- Keep nnU-Net’s stochastic sampling behavior, but make sampled patches prompt-aware.

Do:
1. Add a new dataloader variant in `nnunetv2/training/dataloading/`.
2. Use stochastic patch sampling (not exhaustive sliding).
3. Per sample, select mode with probabilities from `cfg.sampling.mode_probs`:
$$
P(\text{pos})=p_1,\ P(\text{pos+spurious})=p_2,\ P(\text{pos+no-prompt})=p_3,\ P(\text{negative})=p_4,\ \sum_i p_i=1
$$
4. Build patch image, patch target, and prompt channel for that patch.
5. Concatenate prompt channel to image channels.

Strict mode definitions (authoritative, no prior context required):
- Let \(\Omega\) be patch voxels, \(y:\Omega\to\{0,1\}\) GT mask, \(L=\{v\in\Omega\mid y(v)=1\}\), \(B=\Omega\setminus L\).
- Let \(C_\Omega\) be GT lesion centroids that fall inside \(\Omega\).
- Let \(P\) be prompt points for this patch; prompt channel is encoded from \(P\).

1. `pos`:
- Condition: \(|L|>0\).
- Prompts: \(P=\{\text{perturb}(c) : c \in C_\Omega\}\), where \(\text{perturb}\) samples from the propagation error distribution (anisotropic Gaussian offset, truncated).
- If \(C_\Omega=\varnothing\): set \(P=\{\text{perturb}(v^*)\}\), where \(v^*\) is one random voxel from \(L\) (fallback positive point).
- Meaning: simulated propagated prompt(s); training matches longitudinal inference where prompts come from registration propagation.

2. `pos+spurious`:
- Condition: same positive patch as `pos` (\(|L|>0\)).
- Prompts: \(P=\{\text{perturb}(c) : c \in C_\Omega\}\cup S\), with \(|S|=n_{\text{spur}}\), where \(n_{\text{spur}}=\texttt{cfg.sampling.n\_spur}\).
- `spurious` definition: each \(s\in S\) is sampled uniformly from \(B\) (non-lesion voxels), so it is an intentionally wrong positive prompt.
- Meaning: simulate registration/click noise while keeping true prompt(s) present.

3. `pos+no-prompt`:
- Condition: same positive patch as `pos` (\(|L|>0\)).
- Prompts: \(P=\varnothing\), so prompt channel is all zeros.
- `no-prompt` definition: model gets no coordinate prior at all for this positive patch.
- Meaning: force model to segment from image evidence alone when prompt signal is missing.

4. `negative`:
- Condition: \(|L|=0\) (patch contains no lesion voxels).
- Prompts: \(P=S_{\text{neg}}\), with \(|S_{\text{neg}}|=n_{\text{neg}}\), where \(n_{\text{neg}}=\texttt{cfg.sampling.n\_neg}\), sampled uniformly from \(\Omega\).
- `negative` definition: no lesion in target, but prompt points are present and therefore wrong by construction.
- Meaning: teach model to reject false prompts and avoid prompt-driven hallucinations.

Sampling rule:
$$
m \sim \operatorname{Categorical}(p_1, p_2, p_3, p_4)
$$
where \(m \in \{\text{pos},\ \text{pos+spurious},\ \text{pos+no-prompt},\ \text{negative}\}\).

Check:
- `nnunetv2/training/dataloading/data_loader.py`
- `temp/LesionLocator/lesionlocator/utilities/prompt_handling/prompt_handler.py`
- `temp/nnInteractive/nnInteractive/interaction/point.py`

### 5) Large-Lesion Add-On (sparse, not full sliding)
High-level:
- Add bounded extra samples for lesions larger than one patch to avoid truncation bias.

Do:
1. Detect large lesions by bbox-vs-patch size.
2. For each large lesion, add sparse extra positive patches.

Trigger:
$$
\exists k \in \{z,y,x\}: \Delta_k > P_k
$$
where \(\Delta_k\) is lesion bbox size and \(P_k\) patch size in tensor order.

Sampling:
1. Candidate centers from coarse grid inside lesion bbox.
2. Grid stride:
$$
\text{stride}_k = \left\lfloor\frac{P_k}{2}\right\rfloor
$$
3. Keep candidates inside lesion mask.
4. Sample \(K\) centers uniformly from candidates, with \(K=\texttt{cfg.sampling.large\_lesion.K}\), constrained by `K_min/K_max` from config.
5. Cap extras per case per epoch with `cfg.sampling.large_lesion.max_extra`.

Check:
- `temp/LesionLocator/lesionlocator/modules/tracknet.py`

### 6) Trainer Variant
High-level:
- Wire the new dataloader into training while keeping nnU-Net optimization/loss behavior.

Do:
1. Add trainer variant in `nnunetv2/training/nnUNetTrainer/variants/`.
2. Wire new dataloader variant in `get_dataloaders`.
3. Keep standard nnU-Net Dice+CE loss.

Check:
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
- MangoTree `nnunet_dice_ce_loss`

### 7) ROI-Only Inference Entrypoint
High-level:
- LesionLocator-style: full image, single merged prompt, prompt-aware sliding over dilated bbox only.

Do:
1. Add inference module in `nnunetv2/inference/`.
2. Parse propagated points.
3. Build one prompt heatmap for all points in full image shape.
4. Concatenate prompt to image, pad if needed.
5. Run prompt-aware sliding window over dilated bbox region; merge logits (Gaussian weighting).

Prompt-aware slicers (LesionLocator logic, with dilated bbox):
1. Compute prompt bbox from nonzero voxels in `dense_prompt[0]`.
2. Dilate bbox by `patch_size/2` per axis, clamp to image bounds.
3. If dilated bbox extent < patch size in all axes: one centered patch.
4. Else: standard sliding candidates, keep only windows overlapping dilated bbox.

Hard rule:
- In ROI mode, do not call full-volume slicer generation.

Check:
- `nnunetv2/inference/predict_from_raw_data.py`
- `temp/LesionLocator/lesionlocator/inference/lesionlocator_segment.py`
- MangoTree `lesionlocator_prompt_aware_sliding_window_slicers`
- MangoTree `totalsegmentator_multipart_inference`

### 8) CLI Contract
High-level:
- Expose ROI mode with explicit point-space semantics to prevent xyz/zyx ambiguity.

Do:
1. Add ROI-mode CLI entrypoint with:
- `-i/--input`
- `-o/--output`
- `-m/--model_folder`
- `--config` (required path to YAML/JSON config)
- `--points_json`
- `--points_space` (optional override; default from JSON or "voxel")
- `--disable_tta`
2. Parse config first, apply CLI overrides second, route resolved config to Step 7 inference path.

Check:
- `nnunetv2/inference/predict_from_raw_data.py`
- `temp/LesionLocator/lesionlocator/inference/lesionlocator_segment.py`

### 9) Propagated Prompt Simulation
High-level:
- Replace perfect COG prompts with simulated propagated prompts for longitudinal inference.

Do:
1. Add `apply_propagation_offset` in `nnunetv2/utilities/propagated_prompt_simulation.py`.
2. Sample offset from anisotropic Gaussian N(0, diag(σ_z², σ_y², σ_x²)), truncate magnitude to `max_vox`, clip to patch bounds.
3. In pos and pos+spurious modes, apply offset to each centroid before encoding.
4. Add `sampling.propagated` config: `sigma_per_axis`, `max_vox` (defaults from longitudinal COG analysis).

**Default parameters** (from [scripts/cog_propagation_analysis_report.md](../scripts/cog_propagation_analysis_report.md), longitudinal CT dataset: cog_propagated vs cog_fu):
- `sigma_per_axis`: [σ_z, σ_y, σ_x] = [2.75, 5.19, 5.40] — per-axis mean absolute offset in voxels. z is smaller (slice direction) because registration typically preserves axial alignment; x,y have larger in-plane drift.
- `max_vox`: 34.0 — 95th percentile of observed propagation error. Offsets with magnitude > max_vox are rescaled to max_vox (direction preserved). Caps extreme outliers (max observed ≈ 204 vox).
- The distribution naturally yields: ~25% within 1.4 vox (good prompts), median 3.7 vox, mean 9.25 vox (typical propagated), long tail up to 95th.

Check:
- [scripts/cog_propagation_analysis_report.md](../scripts/cog_propagation_analysis_report.md)
- [nnunetv2/utilities/propagated_prompt_simulation.py](../nnunetv2/utilities/propagated_prompt_simulation.py)
