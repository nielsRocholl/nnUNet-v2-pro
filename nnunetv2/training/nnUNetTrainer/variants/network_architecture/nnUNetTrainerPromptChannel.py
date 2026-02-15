from typing import Union, Tuple, List
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerPromptChannel(nnUNetTrainer):
    """Adds 1 extra input channel for prompt heatmap (early concatenation, nnInteractive-style)."""

    PROMPT_CHANNELS = 1

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + nnUNetTrainerPromptChannel.PROMPT_CHANNELS,
            num_output_channels,
            enable_deep_supervision)
