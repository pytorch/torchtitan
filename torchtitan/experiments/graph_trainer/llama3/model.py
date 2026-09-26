# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.llama3 import Llama3Model

from ..model import GraphTrainerModel


class GraphTrainerLlama3Model(GraphTrainerModel, Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def parallelize(self, *, compile_config, **kwargs):
        if compile_config.enable_autoparallel:
            from .parallelize_autoparallel import parallelize_autoparallel_llama

            return parallelize_autoparallel_llama(
                self,
                compile_config=compile_config,
                **kwargs,
            )
        return super().parallelize(
            compile_config=compile_config,
            **kwargs,
        )
