"""Trainer variant: prompt-aware dataloader + prompt channel. Keeps nnU-Net Dice+CE loss."""
from typing import List

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.prompt_aware_data_loader import (
    MODE_NEG,
    MODE_POS,
    MODE_POS_NO_PROMPT,
    MODE_POS_SPUR,
    nnUNetPromptAwareDataLoader,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerPromptChannel import (
    nnUNetTrainerPromptChannel,
)
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.roi_config import DEFAULT_CONFIG_PATH, load_config


class nnUNetTrainerPromptAware(nnUNetTrainerPromptChannel):
    """Wires prompt-aware dataloader into training. Uses bundled default config when config_path is not provided."""

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
            config_path = str(DEFAULT_CONFIG_PATH)
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
        for name in ("pos", "pos_spur", "pos_no_prompt", "neg"):
            self.logger.my_fantastic_logging[f"val_Dice_{name}"] = []

    def _prepare_validation_data(self, data: torch.Tensor) -> torch.Tensor:
        """Add 2 zero prompt channels (pos + neg) for validation / perform_actual_validation."""
        prompt_ch = torch.zeros(2, *data.shape[1:], device=data.device, dtype=data.dtype)
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

    def validation_step(self, batch: dict) -> dict:
        if "mode" not in batch:
            return super().validation_step(batch)
        from nnunetv2.utilities.helpers import autocast, dummy_context

        data = batch["data"]
        target = batch["target"]
        mode = batch["mode"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes_per_sample = list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float16
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target = target.clone()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = (
                    ~target[:, -1:]
                    if target.dtype == torch.bool
                    else 1 - target[:, -1:]
                )
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes_per_sample, mask=mask
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[:, 1:]
            fp_hard = fp_hard[:, 1:]
            fn_hard = fn_hard[:, 1:]

        mode_np = mode if isinstance(mode, np.ndarray) else np.asarray(mode)
        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
            "mode": mode_np,
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]) -> None:
        outputs_collated = collate_outputs(val_outputs)
        if "mode" not in outputs_collated:
            super().on_validation_epoch_end(val_outputs)
            return

        mode_flat = outputs_collated["mode"].reshape(-1)
        tp_all = outputs_collated["tp_hard"]
        fp_all = outputs_collated["fp_hard"]
        fn_all = outputs_collated["fn_hard"]
        if tp_all.ndim == 3:
            n_samples = tp_all.shape[0] * tp_all.shape[1]
            tp_flat = tp_all.reshape(-1, tp_all.shape[-1])
            fp_flat = fp_all.reshape(-1, fp_all.shape[-1])
            fn_flat = fn_all.reshape(-1, fn_all.shape[-1])
        else:
            n_samples = tp_all.shape[0]
            tp_flat = tp_all
            fp_flat = fp_all
            fn_flat = fn_all

        loss_here = np.mean(outputs_collated["loss"])
        self.logger.log("val_losses", loss_here, self.current_epoch)

        mode_names = {
            MODE_POS: "pos",
            MODE_POS_SPUR: "pos_spur",
            MODE_POS_NO_PROMPT: "pos_no_prompt",
            MODE_NEG: "neg",
        }
        for m in (MODE_POS, MODE_POS_SPUR, MODE_POS_NO_PROMPT, MODE_NEG):
            idx = mode_flat == m
            if not np.any(idx):
                self.logger.log(f"val_Dice_{mode_names[m]}", float("nan"), self.current_epoch)
                continue
            tp_m = np.sum(tp_flat[idx], axis=0)
            fp_m = np.sum(fp_flat[idx], axis=0)
            fn_m = np.sum(fn_flat[idx], axis=0)
            dc = np.array(
                [2 * t / (2 * t + p + n) if (2 * t + p + n) > 0 else np.nan for t, p, n in zip(tp_m, fp_m, fn_m)]
            )
            mean_dc = np.nanmean(dc)
            self.logger.log(f"val_Dice_{mode_names[m]}", mean_dc, self.current_epoch)

        tp_global = np.sum(tp_flat, axis=0)
        fp_global = np.sum(fp_flat, axis=0)
        fn_global = np.sum(fn_flat, axis=0)
        dc_global = np.array(
            [2 * t / (2 * t + p + n) if (2 * t + p + n) > 0 else np.nan for t, p, n in zip(tp_global, fp_global, fn_global)]
        )
        mean_fg_dice = np.nanmean(dc_global)
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", dc_global.tolist(), self.current_epoch)

        if self.wandb is not None:
            wb_dict = {}
            for m, name in mode_names.items():
                if f"val_Dice_{name}" in self.logger.my_fantastic_logging:
                    wb_dict[f"val_Dice_{name}"] = self.logger.my_fantastic_logging[f"val_Dice_{name}"][-1]
            if wb_dict:
                self.wandb.log(wb_dict, step=self.current_epoch)
