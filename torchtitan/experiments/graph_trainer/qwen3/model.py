# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.qwen3 import Qwen3Model

from ..model import GraphTrainerModel


class GraphTrainerQwen3Model(GraphTrainerModel, Qwen3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
