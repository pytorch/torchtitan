# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_flex_attn,
)

from . import model_registry


def graph_trainer_deepseek_v3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_flex_attn() -> (GraphTrainer.Config):
    config = to_graph_trainer_config(deepseek_v3_debugmodel_flex_attn(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_16b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_16b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_671b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_671b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config
