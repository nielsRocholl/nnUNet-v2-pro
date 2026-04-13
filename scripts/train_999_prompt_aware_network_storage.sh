#!/bin/bash
# Train Dataset999 with nnUNetTrainerPromptAware — preprocessed data stays on the mounted storage
# server (no rclone/copy to local scratch). Expect more disk latency vs node-local NVMe; if GPU util
# drops or the node I/O pegs, lower nnUNet_n_proc_DA (see documentation/pro/cluster_training_runbook.md).
#
# Shared-node note: nnUNet_n_proc_DA is augmenter worker processes per GPU (~N train + N/2 val).
# Hostname superh200 defaults to 48 when unset — always set explicitly for predictable load.
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=07-00:00:00
#SBATCH --job-name=nnunet-train-999-net
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/train_999_net_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/train_999_net_%j.err
#SBATCH --gres=gpu:1
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export nnUNet_raw=/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed
export nnUNet_results=/nnunet_data/nnUNet_results

export nnUNet_n_proc_DA=16

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

echo "Pre-flight: validating nnUNetv2_train..."
if ! nnUNetv2_train --help &>/dev/null; then
  echo "FATAL: nnUNetv2_train not found or broken."
  exit 1
fi

DS_PREP="${nnUNet_preprocessed}/Dataset999_Merged"
if [[ ! -d "$DS_PREP" ]]; then
  echo "FATAL: preprocessed dataset missing: $DS_PREP"
  exit 1
fi
echo "Using preprocessed data at $DS_PREP (no local copy)."

TRAIN_CMD=(
  nnUNetv2_train
  999
  3d_fullres
  0
  -tr nnUNetTrainerPromptAware
  -p nnUNetResEncUNetLPlans_h200
  --use-wandb
  --wandb-project "nnunet-pro-999"
  --epochs 2500
  --lr-schedule stretched_tail_poly
  --c
  -device cuda
)
echo "Running: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
EXIT=$?
if [[ $EXIT -ne 0 ]]; then
  echo "Fold 0 failed. Use --c to resume."
  exit 1
fi
echo "Fold 0 completed."
