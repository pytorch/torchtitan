# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import BlockMask

from torchtitan.components.tokenizer import BaseTokenizer

from torchtitan.config import JobConfig
from torchtitan.models.attention import VarlenMetadata


AttentionMasksType = dict[str, BlockMask] | BlockMask | VarlenMetadata


@dataclass
class BaseModelArgs:
    """All ModelArgs should inherit from this class.

    The only usage of this class is type checking but allows us to extend common
    arguments to all models in the future.
    """

    _enforced: str = "This field is used to enforce all fields have defaults."

    @abstractmethod
    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        pass

    @abstractmethod
    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        pass


class ModelProtocol(Protocol):
    """Defines the interface for a model class.

    This is used to enforce that all model classes have some methods that are
    required by the trainer.
    """

    def __init__(self, model_args: BaseModelArgs) -> None:
        pass

    @abstractmethod
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights.

        Args:
            buffer_device: Optional device to place buffers on during initialization.
        """
        pass

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        raise NotImplementedError(
            "This model does not support attention masking/Flex Attention."
        )
