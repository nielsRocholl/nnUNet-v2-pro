#!/bin/bash
#SBATCH --qos=high
#SBATCH --exclude=dlc-groudon,dlc-arceus,dlc-slowpoke,dlc-meowth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=07-00:00:00
#SBATCH --job-name=nnunet-plan-pp-999
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/plan_preprocess_999_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/plan_preprocess_999_%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export nnUNet_raw=/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed
export nnUNet_results=/nnunet_data/nnUNet_results

# Config for size_bins (percentile mode) when preprocessing merged datasets.
# Priority: 1) /nnunet_data/nnunet_pro_config.json  2) bundled package config  3) omit (fixed thresholds)
if [ -f /nnunet_data/nnunet_pro_config.json ]; then
  CONFIG_PATH=/nnunet_data/nnunet_pro_config.json
else
  CONFIG_PATH=$(python -c "
from pathlib import Path
try:
    from nnunetv2.utilities.roi_config import DEFAULT_CONFIG_PATH
    p = Path(DEFAULT_CONFIG_PATH)
    print(p) if p.exists() else print('')
except Exception:
    print('')
" 2>/dev/null || true)
fi

# Full merge + plan + preprocess (uncomment to run from scratch):
nnUNetv2_plan_and_preprocess -d 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 --merge -o 999 \
    -pl nnUNetPlannerResEncL_torchres \
    -gpu_memory_target 80 \
    -overwrite_plans_name nnUNetResEncUNetLPlans_h200 \
    -c 3d_fullres \
    -npfp 16 -np 4 \
    --verify_dataset_integrity --reject_failing_cases --verbose \
    ${CONFIG_PATH:+--config "$CONFIG_PATH"}

# Resume after OOM or interruption (fewer workers):
# nnUNetv2_plan_and_preprocess -d 999 \
#   -pl nnUNetPlannerResEncL_torchres \
#   -gpu_memory_target 80 \
#   -overwrite_plans_name nnUNetResEncUNetLPlans_h200 \
#   -c 3d_fullres -npfp 16 -np 4 --resume --verbose \
#   ${CONFIG_PATH:+--config "$CONFIG_PATH"}
