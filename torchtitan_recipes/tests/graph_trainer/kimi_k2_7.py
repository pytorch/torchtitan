# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.kimi_k2_7 import GraphTrainerKimiK2Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.tests.models.kimi_k2_7 import moonlight_16b_a3b


def graph_trainer_moonlight_16b_a3b_adamw(
    seq_len: int | None = None,
) -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        moonlight_16b_a3b(seq_len=seq_len), GraphTrainerKimiK2Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
    )
    return config
