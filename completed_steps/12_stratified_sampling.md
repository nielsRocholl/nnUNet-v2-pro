# Step 12: Stratified Batch Sampling

Stratified batch sampling by (dataset, size_bin) for merged datasets. Add-on classes inherit from nnUNetDataLoader and nnUNetPromptAwareDataLoader; original nnUNet code unchanged.

## Changes

**New:**
- [nnunetv2/training/dataloading/stratified_sampling.py](../nnunetv2/training/dataloading/stratified_sampling.py) — `build_strata`, `build_stratum_weights`, `sample_batch`
- [nnunetv2/training/dataloading/stratified_data_loader.py](../nnunetv2/training/dataloading/stratified_data_loader.py) — `nnUNetStratifiedDataLoader`, `nnUNetPromptAwareStratifiedDataLoader`
- [nnunetv2/training/nnUNetTrainer/variants/sampling/nnUNetTrainerPromptAwareStratified.py](../nnunetv2/training/nnUNetTrainer/variants/sampling/nnUNetTrainerPromptAwareStratified.py) — trainer for prompt-aware + stratified
- [documentation/pro/stratified_sampling_guide.md](../documentation/pro/stratified_sampling_guide.md) — instruction guide
- [tests/test_stratified_sampling.py](../tests/test_stratified_sampling.py) — unit + integration tests

**Modified:**
- [nnunetv2/utilities/dataset_statistics.py](../nnunetv2/utilities/dataset_statistics.py) — `_trim_extremes`, `compute_size_bin_thresholds`, percentile size bins
- [nnunetv2/utilities/roi_config.py](../nnunetv2/utilities/roi_config.py) — `SizeBinsConfig`, `StratifiedConfig`
- [nnunetv2/experiment_planning/plan_and_preprocess_api.py](../nnunetv2/experiment_planning/plan_and_preprocess_api.py) — `--config` for size_bins
- [documentation/pro/multi_dataset_merge.md](../documentation/pro/multi_dataset_merge.md) — Stratified training, link to guide

## Behavior

- **Size bins**: Fixed `[100, 2000, 20000]` or percentile-based (trim extremes, compute quartiles). Config: `size_bins` in nnunet_pro_config.json.
- **Stratified sampler**: `dataset_weights` and `size_bin_weights` control batch makeup. Config: `stratified` in nnunet_pro_config.json. Omit for uniform round-robin.
- Preprocess with `--config` to use percentile size bins; train with `--config` for weighted sampling.

## Usage

```bash
nnUNetv2_plan_and_preprocess -d 1 2 --merge -o 999 -c 3d_fullres --config path/to/nnunet_pro_config.json
nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerPromptAwareStratified -p nnUNetResEncUNetLPlans --config path/to/nnunet_pro_config.json
```

## Tests

```bash
pytest tests/test_stratified_sampling.py tests/test_dataset_statistics.py
```

Integration (slow, requires KiTS23): `pytest tests/test_stratified_sampling.py -m slow`

## References

- [stratified_sampling_guide.md](../documentation/pro/stratified_sampling_guide.md)
- Step 11: [11_dataset_statistics_collection.md](11_dataset_statistics_collection.md)
