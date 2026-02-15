"""Trainer variant: prompt-aware dataloader + prompt channel. Keeps nnU-Net Dice+CE loss."""
from typing import List

import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerPromptChannel import (
    nnUNetTrainerPromptChannel,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.roi_config import load_config


class nnUNetTrainerPromptAware(nnUNetTrainerPromptChannel):
    """Wires prompt-aware dataloader into training. Requires config_path."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device=None,
        display=None,
        use_wandb: bool = None,
        wandb_project: str = None,
        wandb_entity: str = None,
        wandb_run_name: str = None,
        wandb_tags: List[str] = None,
        config_path: str = None,
    ):
        if config_path is None or config_path == "":
            raise ValueError("config_path is required for nnUNetTrainerPromptAware")
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            device=device,
            display=display,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_tags=wandb_tags,
            config_path=config_path,
        )
        self.config_path = config_path
        self.roi_cfg = load_config(config_path)

    def _prepare_validation_data(self, data: torch.Tensor) -> torch.Tensor:
        """Add zero prompt channel for validation (model expects image + prompt)."""
        prompt_ch = torch.zeros(1, *data.shape[1:], device=data.device, dtype=data.dtype)
        return torch.cat([data, prompt_ch], dim=0)

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetPromptAwareDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            cfg=self.roi_cfg,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            force_zero_prompt=False,
        )
        dl_val = nnUNetPromptAwareDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            cfg=self.roi_cfg,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            force_zero_prompt=not self.roi_cfg.prompt.validation_use_prompt,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
