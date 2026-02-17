#!/bin/bash
# Plan only: create 120GB plans for H200.
#SBATCH --qos=high
#SBATCH --exclude=dlc-groudon,dlc-arceus,dlc-slowpoke,dlc-meowth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=01-00:00:00
#SBATCH --job-name=nnunet-plan-999-120gb
#SBATCH --output=/data/oncology/experiments/nielsrocholl/plan_999_120gb_%j.out
#SBATCH --error=/data/oncology/experiments/nielsrocholl/plan_999_120gb_%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:20260216-pro-5"

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export nnUNet_raw=/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed
export nnUNet_results=/nnunet_data/nnUNet_results

# Overwrites nnUNetResEncUNetLPlans_h200.json with 120GB config (larger batch).
# Existing preprocessed folder nnUNetResEncUNetLPlans_h200_3d_fullres stays valid.
nnUNetv2_plan_experiment -d 999 -pl nnUNetPlannerResEncL_torchres \
  -gpu_memory_target 120 \
  -overwrite_plans_name nnUNetResEncUNetLPlans_h200 \
  --verbose
