# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 GraphTrainer configurations used by tests."""

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)

from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.tests.models.qwen3 import qwen3_debugmodel, qwen3_moe_debug


def graph_trainer_qwen3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_debugmodel(), GraphTrainerQwen3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_qwen3_debugmodel_moe() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_moe_debug(), GraphTrainerQwen3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    config.parallelism.expert_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config
