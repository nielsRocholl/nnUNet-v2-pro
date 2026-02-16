#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --job-name=nnunet-plan-pp-999
#SBATCH --output=/data/oncology/experiments/nielsrocholl/plan_preprocess_999_%j.out
#SBATCH --error=/data/oncology/experiments/nielsrocholl/plan_preprocess_999_%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:20260216-pro-2"
#SBATCH --exclude=dlc-groudon,dlc-arceus,dlc-slowpoke,dlc-meowth

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

# nnUNet paths (adjust if your structure differs)
export nnUNet_raw=/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed
export nnUNet_results=/nnunet_data/nnUNet_results

nnUNetv2_plan_and_preprocess -d 10 11 12 --merge -o 999 \
    -pl nnUNetPlannerResEncL_torchres \
    -gpu_memory_target 80 \
    -overwrite_plans_name nnUNetResEncUNetLPlans_h200 \
    -c 3d_fullres \
    -npfp 48 -np 48 \
    --verify_dataset_integrity --reject_failing_cases --verbose
