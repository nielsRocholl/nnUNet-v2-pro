from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprint_dataset
from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset
from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess_dataset
from nnunetv2.run.run_training import run_training
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.cli_display import InferenceDisplay
from batchgenerators.utilities.file_and_folder_operations import join
import argparse
import torch
import os

if __name__ == '__main__':
    # import step from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, required=True)
    args = parser.parse_args()
    step = args.step

    if step == 'fingerprint':
        # Test fingerprint extraction with new CLI display
        fingerprint = extract_fingerprint_dataset(
            dataset_id=47,
            check_dataset_integrity=True,
            clean=True,
            verbose=False  # Set to True to see verbose output
        )
        print(f"\nFingerprint extracted successfully")
    elif step == 'plan':
        # Test experiment planning with new CLI display
        plans, plans_identifier = plan_experiment_dataset(
            dataset_id=47,
            verbose=False  # Set to True to see verbose output
        )
        print(f"\nPlans identifier: {plans_identifier}")
    elif step == 'preprocess':
        # Test preprocessing with new CLI display
        preprocess_dataset(
            dataset_id=47,
            verbose=False  # Set to True to see verbose output
        )
        print(f"\nPreprocessing completed successfully")
    elif step == 'plan_and_preprocess':
        # Test combined fingerprint extraction, planning, and preprocessing
        # This mimics nnUNetv2_plan_and_preprocess command
        fingerprint = extract_fingerprint_dataset(
            dataset_id=47,
            check_dataset_integrity=True,
            clean=True,
            verbose=False
        )
        plans, plans_identifier = plan_experiment_dataset(
            dataset_id=47,
            verbose=False
        )
        preprocess_dataset(
            dataset_id=47,
            plans_identifier=plans_identifier,
            verbose=False
        )
        print(f"\nPlan and preprocess completed successfully")
    elif step == 'train':
        # Test training with default output (will improve CLI display later)
        # Fix macOS multiprocessing issues by disabling data augmentation workers
        import os
        import multiprocessing
        
        # Set correct environment variable (note: it's nnUNet_n_proc_DA, not nnUNet_def_n_proc_DA)
        os.environ['nnUNet_n_proc_DA'] = '0'  # Disable multiprocessing to avoid macOS shared memory issues
        
        # Disable multiprocessing for validation/inference as well
        os.environ['nnUNet_def_n_proc'] = '0'  # Single process for validation/inference
        
        # Also set PyTorch multiprocessing start method to 'spawn' (required for macOS)
        multiprocessing.set_start_method('spawn', force=True)
        
        # Try MPS first (Apple Silicon), fallback to CPU if not available
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device('cpu')
            print("MPS not available, using CPU")
        
        run_training(
            dataset_name_or_id='47',  # Pass as string - function expects string or Dataset name
            configuration='3d_fullres',
            fold=0,
            device=device,
            disable_checkpointing=False  # Keep checkpointing so we have a model for inference
        )
        print(f"\nTraining completed successfully")
    elif step == 'inference':
        # Test inference with trained model
        import multiprocessing
        
        # Fix macOS multiprocessing issues
        os.environ['nnUNet_def_n_proc'] = '0'  # Single process for inference
        multiprocessing.set_start_method('spawn', force=True)
        
        # Try MPS first (Apple Silicon), fallback to CPU if not available
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device('cpu')
            print("MPS not available, using CPU")
        
        dataset_id = 47
        configuration = '3d_fullres'
        fold = 0
        trainer_name = 'nnUNetTrainer'
        plans_identifier = 'nnUNetPlans'
        
        # Get model folder path (base folder, without fold - initialize_from_trained_model_folder handles fold internally)
        model_folder = get_output_folder(dataset_id, trainer_name, plans_identifier, configuration, fold=None)
        dataset_name = convert_id_to_dataset_name(dataset_id)
        
        # Input folder (use validation images from training set for testing)
        input_folder = join(nnUNet_raw, dataset_name, 'imagesTr')
        
        # Output folder for predictions
        output_folder = join(nnUNet_raw, dataset_name, 'predictions')
        os.makedirs(output_folder, exist_ok=True)
        
        # Create predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=False,  # False for MPS/CPU
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        # Initialize from trained model
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(fold,),
            checkpoint_name='checkpoint_final.pth'
        )
        
        # Count cases using the same logic as predict_from_files_sequential
        from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder
        num_cases = len(get_identifiers_from_splitted_dataset_folder(
            input_folder,
            predictor.dataset_json['file_ending']
        ))
        
        # Run inference with display
        device_str = "mps" if device.type == "mps" else "cuda" if device.type == "cuda" else "cpu"
        with InferenceDisplay(dataset_name, configuration, device_str, num_cases, verbose=False) as display:
            predictor.predict_from_files_sequential(
                input_folder,
                output_folder,
                save_probabilities=False,
                overwrite=True,
                folder_with_segs_from_prev_stage=None,
                display=display
            )