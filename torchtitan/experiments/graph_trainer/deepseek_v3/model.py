# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.models.deepseek_v3 import DeepSeekV3Model

from ..simple_fsdp import disable_active_parametrization


class GraphTrainerDeepSeekV3Model(DeepSeekV3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)
