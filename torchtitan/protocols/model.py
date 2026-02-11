# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Module
from torchtitan.config.configurable import Configurable
from torchtitan.models.attention import VarlenMetadata


AttentionMasksType = dict[str, BlockMask] | BlockMask | VarlenMetadata


class BaseModel(Module):
    """Base class for all model classes.

    Models inherit from BaseModel (which is Module = nn.Module + Configurable).
    Each model defines a nested Config(BaseModel.Config) with model hyperparameters.
    The model is constructed via ``config.build()``.

    All models must implement ``init_weights`` (from Module).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Base config for all models.

        Subclasses define model-specific hyperparameters.
        """

        # TODO(lty): This function violates encapsulation;
        # maybe replace it with config passes from outside.
        @abstractmethod
        def update_from_config(
            self,
            *,
            job_config,
            **kwargs,
        ) -> None:
            pass

        @abstractmethod
        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
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
