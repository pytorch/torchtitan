# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.deepseek_v3 import DeepSeekV3Model, DeepSeekV3ModelArgs

from ..simple_fsdp import disable_active_parametrization


class SimpleFSDPDeepSeekV3Model(DeepSeekV3Model):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__(model_args)
        self.init_weights()

    def init_weights(self, *args, **kwargs):
        with disable_active_parametrization():
            super().init_weights(*args, **kwargs)
