# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import JobConfig


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
