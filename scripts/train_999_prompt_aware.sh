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
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:20260216-pro-5"

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export nnUNet_raw=/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed
export nnUNet_results=/nnunet_data/nnUNet_results

# Prompt-aware config (EDT encoding, robust to misaligned prompts)
CONFIG_PATH=/nnunet_data/nnunet_pro_config.json
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Creating nnunet_pro_config.json at $CONFIG_PATH"
  cat > "$CONFIG_PATH" << 'EOF'
{"prompt":{"point_radius_vox":5,"encoding":"edt","validation_use_prompt":true},"sampling":{"mode_probs":[0.5,0.2,0.15,0.15],"n_spur":[1,2],"n_neg":[1,3],"large_lesion":{"K":2,"K_min":1,"K_max":4,"max_extra":3}},"inference":{"tile_step_size":0.75,"disable_tta_default":false}}
EOF
fi

# Data loading workers (tune if needed; 12–16 often good for large datasets)
export nnUNet_n_proc_DA=16

# Train all 5 folds for strongest ensemble
# --npz: save softmax for find_best_configuration / ensemble
# -p: plans name from preprocessing (nnUNetResEncUNetLPlans_h200)
for fold in 0 1 2 3 4; do
  echo "=========================================="
  echo "Training fold $fold"
  echo "=========================================="
  nnUNetv2_train 999 3d_fullres $fold \
    -tr nnUNetTrainerPromptAware \
    -p nnUNetResEncUNetLPlans_h200 \
    --config "$CONFIG_PATH" \
    --npz \
    -device cuda
  if [[ $? -ne 0 ]]; then
    echo "Fold $fold failed. Use --c to resume: nnUNetv2_train 999 3d_fullres $fold ... --c"
    exit 1
  fi
done

echo "All 5 folds completed."
