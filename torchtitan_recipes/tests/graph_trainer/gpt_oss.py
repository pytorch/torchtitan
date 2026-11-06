# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPT-OSS GraphTrainer configurations used by tests."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.gpt_oss.model import GraphTrainerGptOssModel
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.tests.models.gpt_oss import (
    gpt_oss_debugmodel,
    gpt_oss_debugmodel_flex,
)


def graph_trainer_gpt_oss_debugmodel(
    parallelism: ParallelismConfig | None = None,
) -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        gpt_oss_debugmodel(parallelism=parallelism), GraphTrainerGptOssModel.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_gpt_oss_debugmodel_flex(
    parallelism: ParallelismConfig | None = None,
) -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        gpt_oss_debugmodel_flex(parallelism=parallelism),
        GraphTrainerGptOssModel.Config,
    )
    config.compile = GraphTrainerCompileConfig()
    return config
