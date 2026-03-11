# Step 11: Dataset Statistics Collection

Collect per-case dataset statistics during multi-dataset preprocessing. Enables stratified batch sampling by lesion size and dataset source.

## Changes

**New:**
- [nnunetv2/utilities/dataset_statistics.py](../nnunetv2/utilities/dataset_statistics.py) — `collect_case_statistics`, `save_case_stats`, `get_size_bin`
- [tests/test_dataset_statistics.py](../tests/test_dataset_statistics.py) — unit + integration tests
- [scripts/create_dummy_datasets_for_stats.py](../scripts/create_dummy_datasets_for_stats.py) — create KiTS23-based dummy datasets

**Modified:**
- [nnunetv2/utilities/multi_dataset_merge.py](../nnunetv2/utilities/multi_dataset_merge.py) — add `source_datasets` to merged `dataset.json`
- [nnunetv2/experiment_planning/plan_and_preprocess_api.py](../nnunetv2/experiment_planning/plan_and_preprocess_api.py) — call stats collection after preprocessing when `source_datasets` present
- [tests/conftest.py](../tests/conftest.py) — add `KITS23_NNUNET_RAW`

## Behavior

When preprocessing a **merged** dataset (one with `source_datasets` in `dataset.json`):

1. After each configuration’s preprocessing completes, `collect_case_statistics` runs.
2. For each preprocessed case: loads `_seg.b2nd`, computes connected components via `scipy.ndimage.label`.
3. Per case, extracts:
   - `fg_voxels.total` — total foreground voxels
   - `fg_voxels.min_cc` — smallest connected component size
   - `fg_voxels.max_cc` — largest connected component size
   - `fg_voxels.mean_cc` — mean component size
   - `dataset` — source dataset (from `source_datasets` prefix match)
   - `size_bin` — background/tiny/small/medium/large (from `max_cc` thresholds)
4. Saves `case_stats_{config}.json` in `nnUNet_preprocessed/{dataset_name}/`.

## Output format

```json
{
  "_metadata": {"size_bin_thresholds": [100, 2000, 20000]},
  "Dataset001_Mini_KiTS23_case_00004": {
    "dataset": "Dataset001_Mini",
    "fg_voxels": {"total": 1234, "min_cc": 50, "max_cc": 800, "mean_cc": 308},
    "size_bin": "small"
  }
}
```

## Test

`pytest tests/test_dataset_statistics.py`

Integration (slow, requires KiTS23): `pytest tests/test_dataset_statistics.py::test_dataset_statistics_collection -m slow`

## References

- Plan: [high_level_plan.md](../high_level_plan.md) — Phase 2 Stratified Batch Sampling
- Documentation: [documentation/pro/multi_dataset_merge.md](../documentation/pro/multi_dataset_merge.md)
