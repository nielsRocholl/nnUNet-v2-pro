"""Trainer: prompt-aware + stratified batch sampling. Requires case_stats_{config}.json."""
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
from nnunetv2.training.dataloading.stratified_data_loader import nnUNetPromptAwareStratifiedDataLoader
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPromptAware import nnUNetTrainerPromptAware
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetTrainerPromptAwareStratified(nnUNetTrainerPromptAware):
    """Prompt-aware + stratified. Uses nnUNetPromptAwareStratifiedDataLoader for train."""

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        case_stats_path = join(
            self.preprocessed_dataset_folder_base,
            f"case_stats_{self.configuration_name}.json",
        )
        if not isfile(case_stats_path):
            raise FileNotFoundError(
                f"Stratified trainer requires case_stats: {case_stats_path}. "
                "Run preprocessing on a merged dataset first."
            )

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

        dl_tr = nnUNetPromptAwareStratifiedDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            cfg=self.roi_cfg,
            case_stats_path=case_stats_path,
            stratified_config=self.roi_cfg.stratified,
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
