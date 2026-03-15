export nnUNet_results=/nnunet_data/nnUNet_results

export nnUNet_preprocessed=/nnunet_data/nnUNet_preprocessed

export nnUNet_raw=/nnunet_data/nnUNet_raw

nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerPromptAware -p nnUNetResEncUNetLPlans_h200 --npz --use-wandb --wandb-project "nnunet-pro-999" -device cuda