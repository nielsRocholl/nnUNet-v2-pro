# nnU-Net v2 Pro

**Enhanced nnU-Net v2 with improved CLI experience and experiment tracking**

This is a customized version of nnU-Net v2 with significant enhancements to the user experience and workflow integration. All core nnU-Net functionality remains unchanged, with added features for better usability and experiment management.

## 🚀 What's New in the Pro Version?

### Enhanced CLI Displays
- **Rich-formatted terminal output** with progress bars, time estimates, and structured displays
- **Pipeline flow diagrams** showing the current stage in the processing pipeline
- **Consolidated warnings and messages** for cleaner, more readable output
- **Real-time progress tracking** for fingerprint extraction, planning, preprocessing, training, and inference
- **Unified display system** across all pipeline stages with consistent formatting

### Weights & Biases Integration
- **Optional experiment tracking** with [Weights & Biases](https://wandb.ai)
- **Automatic metric logging** (loss, Dice scores, learning rate, epoch time, etc.)
- **Flexible configuration** via CLI arguments, environment variables, or programmatic API
- **Seamless resume support** - automatically continues tracking when resuming training
- **Clickable links** in terminal output for quick access to your experiments

See the [wandb integration guide](documentation/pro/wandb_integration.md) for detailed usage instructions.

### Improved User Experience
- **Better inference display** - single progress bar for all cases with average time per case
- **Cleaner output** - organized messages and warnings, less clutter
- **Time estimates** - see how long each stage will take

### Multi-Dataset Merge
- **Merge multiple datasets** into one without copying raw files
- Use `--merge` and `-o` with `nnUNetv2_plan_and_preprocess` (or the separate fingerprint/plan/preprocess commands)
- Source datasets must share the same `channel_names`, `file_ending`, and `labels`

See [documentation/pro/multi_dataset_merge.md](documentation/pro/multi_dataset_merge.md) for usage.

### ROI-Prompted Segmentation (Steps 1–8)

A complete prompt-aware extension for lesion segmentation:

- **Prompt channel** — One extra input channel for point/centroid prompts (nnInteractive-style early concatenation)
- **Coordinate handling** — Voxel or world-space points, converted to preprocessed `(z,y,x)`
- **Prompt encoding** — Lesion centroids → heatmap (binary or EDT), merged with `torch.maximum`
- **Prompt-aware training** — Four sampling modes (pos, pos+spurious, pos+no-prompt, negative) to handle noisy/missing prompts
- **Large-lesion sampling** — Extra patches for lesions larger than one patch to reduce truncation bias
- **nnUNetTrainerPromptAware** — Trainer variant wiring the prompt-aware dataloader; keeps standard Dice+CE loss
- **ROI-only inference** — Sliding windows over dilated bbox only (no full-volume sliding)
- **CLI** — `nnUNetv2_predict_roi` with `--config`, `--points_json`, optional `--points_space`

See [documentation/pro/prompt_aware_guide.md](documentation/pro/prompt_aware_guide.md) for detailed usage.

## Getting Started

For installation, dataset conversion, and usage instructions, refer to the [original nnU-Net v2 documentation](https://github.com/MIC-DKFZ/nnUNet).

**Pro Version Features:**
- [ROI-Prompted Segmentation Guide](documentation/pro/prompt_aware_guide.md) - Train and run inference with point prompts
- [Weights & Biases Integration Guide](documentation/pro/wandb_integration.md) - Track your experiments with wandb
- [Multi-Dataset Merge](documentation/pro/multi_dataset_merge.md) - Merge datasets without copying raw files

## Citation

Please cite the following paper when using nnU-Net:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## References

- **Original nnU-Net v2**: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- **Original nnU-Net v1**: [https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)

## Acknowledgements

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
