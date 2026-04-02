"""Stratified dataloaders: override get_indices for (dataset, size_bin) batch composition."""
from typing import List, Optional, Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, load_json

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.prompt_aware_data_loader import nnUNetPromptAwareDataLoader
from nnunetv2.training.dataloading.stratified_sampling import build_strata, build_stratum_weights, sample_batch
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def _get_weights(strata, stratified_config) -> Optional[dict]:
    if stratified_config is None:
        return None
    return build_stratum_weights(
        strata,
        stratified_config.dataset_weights,
        stratified_config.size_bin_weights,
    )


class nnUNetStratifiedDataLoader(nnUNetDataLoader):
    """Stratifies batches by (dataset, size_bin). Requires case_stats_{config}.json."""

    def __init__(
        self,
        data: nnUNetBaseDataset,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        label_manager: LabelManager,
        case_stats_path: str,
        stratified_config=None,
        oversample_foreground_percent: float = 0.0,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling: bool = False,
        transforms=None,
    ):
        super().__init__(
            data, batch_size, patch_size, final_patch_size, label_manager,
            oversample_foreground_percent, sampling_probabilities, pad_sides,
            probabilistic_oversampling, transforms,
        )
        if not isfile(case_stats_path):
            raise FileNotFoundError(f"case_stats required for stratified sampling: {case_stats_path}")
        case_stats = load_json(case_stats_path)
        self._strata = build_strata(case_stats, list(self.indices))
        if not self._strata:
            raise ValueError("No strata after filtering case_stats to training keys")
        self._weights = _get_weights(self._strata, stratified_config)

    def get_indices(self) -> List[str]:
        return sample_batch(self._strata, self.batch_size, weights=self._weights)


class nnUNetPromptAwareStratifiedDataLoader(nnUNetPromptAwareDataLoader):
    """Prompt-aware + stratified. Overrides get_indices only."""

    def __init__(self, data, batch_size, patch_size, final_patch_size, label_manager, cfg, case_stats_path: str,
                 stratified_config=None, oversample_foreground_percent=0.0, sampling_probabilities=None, pad_sides=None,
                 probabilistic_oversampling=False, transforms=None, force_zero_prompt=False):
        super().__init__(
            data, batch_size, patch_size, final_patch_size, label_manager, cfg,
            oversample_foreground_percent, sampling_probabilities, pad_sides,
            probabilistic_oversampling, transforms, force_zero_prompt,
        )
        if not isfile(case_stats_path):
            raise FileNotFoundError(f"case_stats required for stratified sampling: {case_stats_path}")
        case_stats = load_json(case_stats_path)
        self._strata = build_strata(case_stats, list(self.indices))
        if not self._strata:
            raise ValueError("No strata after filtering case_stats to training keys")
        self._weights = _get_weights(self._strata, stratified_config)

    def get_indices(self) -> List[str]:
        return sample_batch(self._strata, self.batch_size, weights=self._weights)
