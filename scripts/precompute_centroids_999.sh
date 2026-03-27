#!/bin/bash
#SBATCH --qos=high
#SBATCH --exclude=dlc-groudon,dlc-arceus,dlc-slowpoke,dlc-meowth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=07-00:00:00
#SBATCH --job-name=nnunet-centroids-999
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/centroids_999_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/centroids_999_%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed

nnUNetv2_precompute_centroids -d 999 -plans_name nnUNetResEncUNetLPlans_h200 -c 3d_fullres -np 24 --resume
