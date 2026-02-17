#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=21-00:00:00
#SBATCH --job-name=nnunet-train-999
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/train_999_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/train_999_%j.err
#SBATCH --gres=gpu:1
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:20260216-pro-6"

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export nnUNet_raw=/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed
export nnUNet_results=/nnunet_data/nnUNet_results

# Data loading workers (tune if needed; 12–16 often good for large datasets)
export nnUNet_n_proc_DA=16

# Train fold 0
# --npz: save softmax for find_best_configuration / ensemble
# -p: plans name from preprocessing (nnUNetResEncUNetLPlans_h200)
nnUNetv2_train 999 3d_fullres 0 \
  -tr nnUNetTrainerPromptAware \
  -p nnUNetResEncUNetLPlans_h200 \
  --npz \
  -device cuda
EXIT=$?
if [[ $EXIT -ne 0 ]]; then
  echo "Fold 0 failed. Use --c to resume."
  exit 1
fi
echo "Fold 0 completed."
