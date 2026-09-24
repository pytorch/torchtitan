# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.deepseek_v3 import DeepSeekV3Model

from ..model import GraphTrainerModel


class GraphTrainerDeepSeekV3Model(GraphTrainerModel, DeepSeekV3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def parallelize(self, *, compile_config, **kwargs):
        if compile_config.enable_autoparallel:
            from .parallelize_autoparallel import parallelize_autoparallel_deepseekv3

            return parallelize_autoparallel_deepseekv3(
                self,
                compile_config=compile_config,
                **kwargs,
            )
        return super().parallelize(
            compile_config=compile_config,
            **kwargs,
        )
