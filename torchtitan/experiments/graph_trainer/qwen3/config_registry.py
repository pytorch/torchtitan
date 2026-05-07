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
from torchtitan.models.qwen3.config_registry import (
    qwen3_14b,
    qwen3_debugmodel,
    qwen3_debugmodel_flex,
    qwen3_moe_debug,
    qwen3_moe_debug_ep,
)

from . import model_registry


def graph_trainer_qwen3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_debugmodel_flex_attn() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_debugmodel_flex(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_debugmodel_moe() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_moe_debug(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_debugmodel_moe_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_moe_debug_ep(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_14b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_14b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


