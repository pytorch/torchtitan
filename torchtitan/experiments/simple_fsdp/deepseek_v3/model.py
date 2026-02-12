# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.deepseek_v3 import DeepSeekV3Model

from ..simple_fsdp import disable_active_parametrization


class SimpleFSDPDeepSeekV3Model(DeepSeekV3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def init_weights(self, *args, **kwargs):
        with disable_active_parametrization():
            super().init_weights(*args, **kwargs)
