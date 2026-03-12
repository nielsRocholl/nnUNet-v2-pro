# Step 13: Extended wandb Metrics

Extended Dice metrics for wandb when using nnUNetTrainerPromptAware or nnUNetTrainerPromptAwareStratified: per-patch, per-dataset, and per-lesion-size-bin.

## Changes

**New:**
- [nnunetv2/utilities/extended_metrics.py](../nnunetv2/utilities/extended_metrics.py) — `compute_extended_dice_metrics`
- [tests/test_extended_metrics.py](../tests/test_extended_metrics.py) — unit + integration tests

**Modified:**
- [nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerPromptAware.py](../nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerPromptAware.py) — validation returns `keys`, `on_validation_epoch_end` computes and logs extended metrics
- [documentation/pro/wandb_integration.md](../documentation/pro/wandb_integration.md) — Extended Metrics section

## Metrics

| Metric | Definition |
|--------|------------|
| `dice_per_patch` | **Patch-based**: mean Dice per patch, then averaged over all patches (proxy for patch-based inference) |
| `dice_dataset_{name}` | **nnUNet-style**: aggregate tp/fp/fn across patches in that dataset, then Dice from totals |
| `dice_size_bin_{bin}` | **nnUNet-style**: aggregate tp/fp/fn across patches in that size bin (tiny, small, medium, large), then Dice from totals |

Per-dataset and per-size-bin require `case_stats_{config}.json` (from Step 11). Vanilla nnUNet trainers unchanged.

## Behavior

- Validation batches carry `keys` (case identifiers) through the pipeline.
- After validation, `compute_extended_dice_metrics` groups patches by dataset and size_bin via case_stats.
- Keys are flattened when collated as numpy arrays (dataloader returns ndarray from `get_indices`).

## Usage

```bash
nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerPromptAware --use-wandb
```

Extended metrics appear in wandb when using PromptAware trainers on merged datasets with case_stats.

## Tests

```bash
pytest tests/test_extended_metrics.py
```

Integration (slow, requires preprocessed Dataset999): `pytest tests/test_extended_metrics.py::test_extended_metrics_training_one_epoch -m slow`

## References

- [wandb_integration.md](../documentation/pro/wandb_integration.md)
- Step 11: [11_dataset_statistics_collection.md](11_dataset_statistics_collection.md)
