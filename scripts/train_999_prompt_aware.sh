#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=07-00:00:00
#SBATCH --job-name=nnunet-train-999
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/train_999_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/train_999_%j.err
#SBATCH --gres=gpu:1
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:pro-9"

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export nnUNet_raw=/nnunet_data/nnUNet_raw
export nnUNet_results=/nnunet_data/nnUNet_results
export nnUNet_n_proc_DA=16

# Copy preprocessed to compute node (parent dir for nnUNet)
LOCAL_PREP=/root/nnUNet_preprocessed
mkdir -p "$LOCAL_PREP"
echo "Copying preprocessed data to compute node..."
rclone copy /nnunet_data/nnUNet_preprocessed/Dataset999_Merged/ "$LOCAL_PREP/Dataset999_Merged" \
  --progress \
  --transfers 48 \
  --multi-thread-streams 16 \
  --no-update-modtime \
  --retries 5 \
  --copy-links
if [[ $? -ne 0 ]]; then
  echo "rclone copy failed."
  exit 1
fi
echo "Copy complete. Verifying..."
n_files=$(find "$LOCAL_PREP/Dataset999_Merged" -name "*.b2nd" | wc -l)
echo "Found $n_files .b2nd files on compute node."

export nnUNet_preprocessed="$LOCAL_PREP"

nnUNetv2_train 999 3d_fullres 0 \
  -tr nnUNetTrainerPromptAware \
  -p nnUNetResEncUNetLPlans_h200 \
  --npz \
  --use-wandb \
  --wandb-project "nnunet-pro-999" \
  -device cuda \
  --c
EXIT=$?
rm -rf "$LOCAL_PREP/Dataset999_Merged"
if [[ $EXIT -ne 0 ]]; then
  echo "Fold 0 failed. Use --c to resume."
  exit 1
fi
echo "Fold 0 completed."