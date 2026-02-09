# Weights & Biases (wandb) Integration Guide

## Installation

Install wandb as an optional dependency:

```bash
pip install wandb
```

Or install with the optional dependency group:

```bash
pip install nnunetv2[wandb]
```

## Setup

### 1. Create a wandb account

Sign up at [wandb.ai](https://wandb.ai) if you don't have an account.

### 2. Login

```bash
wandb login
```

This will prompt you for your API key (found in your wandb account settings).

## Usage

### Method 1: CLI Arguments (Recommended)

Enable wandb directly from the command line:

```bash
nnUNetv2_train Dataset001_MyDataset 3d_fullres 0 --use-wandb
```

With custom project and run name:

```bash
nnUNetv2_train Dataset001_MyDataset 3d_fullres 0 \
    --use-wandb \
    --wandb-project "my-segmentation-project" \
    --wandb-entity "my-team" \
    --wandb-run-name "experiment-1-fold0" \
    --wandb-tags "baseline,3d-fullres"
```

Disable wandb explicitly (overrides environment variables):

```bash
nnUNetv2_train Dataset001_MyDataset 3d_fullres 0 --no-wandb
```

### Method 2: Environment Variables

Set environment variables for global defaults:

```bash
export nnUNet_use_wandb=true
export nnUNet_wandb_project="my-project"
export nnUNet_wandb_entity="my-team"
export nnUNet_wandb_run_name="custom-run-name"
export nnUNet_wandb_tags="tag1,tag2,tag3"

nnUNetv2_train Dataset001_MyDataset 3d_fullres 0
```

### Method 3: Programmatic (Python API)

When using the training API directly (e.g., in `test.py` or custom scripts):

```python
from nnunetv2.run.run_training import run_training
import torch

run_training(
    dataset_name_or_id='47',
    configuration='3d_fullres',
    fold=0,
    device=torch.device('mps'),
    use_wandb=True,
    wandb_project='my-project',
    wandb_entity='my-team',
    wandb_run_name='custom-run',
    wandb_tags=['experiment', 'baseline']
)
```

Or when creating a trainer directly:

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch

trainer = nnUNetTrainer(
    plans=plans,
    configuration='3d_fullres',
    fold=0,
    dataset_json=dataset_json,
    device=torch.device('cuda'),
    use_wandb=True,
    wandb_project='my-project'
)
```

## Configuration Priority

Configuration is resolved in this order (highest priority first):

1. **CLI arguments** (`--use-wandb`, `--wandb-project`, etc.)
2. **Function/trainer parameters** (when using Python API)
3. **Environment variables** (`nnUNet_use_wandb`, `nnUNet_wandb_project`, etc.)
4. **Defaults** (wandb disabled)

## What Gets Logged

### Per Epoch Metrics:
- `train_loss` - Training loss
- `val_loss` - Validation loss
- `mean_fg_dice` - Mean foreground Dice score
- `ema_fg_dice` - Exponential moving average of foreground Dice
- `dice_class_0`, `dice_class_1`, ... - Dice score per class
- `lr` - Learning rate
- `epoch_time` - Time taken for the epoch (seconds)
- `best_ema_dice` - Best EMA Dice score (logged when new best is found)
- `is_best` - Flag indicating if this is the best model so far

### Configuration (logged once at training start):
- `dataset_name` - Dataset identifier
- `configuration` - Configuration name (e.g., "3d_fullres")
- `fold` - Cross-validation fold
- `trainer` - Trainer class name
- `batch_size` - Batch size
- `patch_size` - Patch size used for training
- `spacing` - Image spacing
- `num_epochs` - Total number of epochs
- `initial_lr` - Initial learning rate
- `device` - Device used (cuda/cpu/mps)
- `oversample_foreground_percent` - Foreground oversampling percentage
- `num_iterations_per_epoch` - Training iterations per epoch
- `num_val_iterations_per_epoch` - Validation iterations per epoch

## Resuming Training

When continuing training with `--c` flag, wandb will automatically resume the previous run:

```bash
nnUNetv2_train Dataset001_MyDataset 3d_fullres 0 --c --use-wandb
```

The wandb run ID is saved in checkpoints, so the same run continues logging.

## Distributed Training (DDP)

When using multiple GPUs, wandb logging only happens on rank 0 to avoid duplicate logs.

## Troubleshooting

### wandb not installed

If you enable wandb but haven't installed it, you'll see:

```
WARNING: wandb requested but not installed. Install with: pip install wandb
```

Solution: `pip install wandb`

### Authentication errors

If you see authentication errors, run:

```bash
wandb login
```

### Disable wandb for a specific run

Use `--no-wandb` flag to override environment variables:

```bash
nnUNetv2_train Dataset001_MyDataset 3d_fullres 0 --no-wandb
```

## Examples

### Basic usage with auto-generated run name:

```bash
nnUNetv2_train Dataset001_MyDataset 3d_fullres 0 --use-wandb
```

Run name will be: `Dataset001_MyDataset_3d_fullres_fold0_20250209_143022`

### Custom project and tags:

```bash
nnUNetv2_train Dataset001_MyDataset 3d_fullres 0 \
    --use-wandb \
    --wandb-project "medical-segmentation" \
    --wandb-tags "baseline,fullres,fold0"
```

### Using environment variables for defaults:

```bash
# Set once in your shell
export nnUNet_use_wandb=true
export nnUNet_wandb_project="my-research"

# All subsequent training runs will use wandb
nnUNetv2_train Dataset001_MyDataset 3d_fullres 0
nnUNetv2_train Dataset001_MyDataset 3d_fullres 1
```
