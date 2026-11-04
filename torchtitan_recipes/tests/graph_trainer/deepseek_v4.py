# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4 GraphTrainer configurations used by tests."""

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v4.model import (
    GraphTrainerDeepSeekV4Model,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel


def deepseek_v4_debugmodel_adamw(
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """DeepSeek V4 debugmodel with AdamW.

    SimpleFSDP parameters do not carry the FlexShard layouts DistMuon reads, so
    AdamW trains every parameter.
    """
    config = deepseek_v4_debugmodel(parallelism=parallelism)
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    return config


def graph_trainer_deepseek_v4_debugmodel(
    parallelism: ParallelismConfig | None = None,
) -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v4_debugmodel_adamw(parallelism=parallelism),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = GraphTrainerCompileConfig()
    return config
