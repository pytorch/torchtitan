# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.qwen3.config_registry import (
    qwen3_debugmodel,
    qwen3_debugmodel_flex,
    qwen3_moe_debug,
    qwen3_moe_debug_ep,
)
from torchtitan.trainer import Trainer

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


# CrossEntropyLoss baselines for graph_trainer numerics tests. graph_trainer
# doesn't yet support ChunkedCELoss, so to_graph_trainer_config swaps it for
# CrossEntropyLoss; these wrappers apply the same swap to the eager baseline
# so loss_compare runs apples-to-apples.
# TODO: Remove once graph_trainer supports ChunkedCELoss.
def qwen3_debugmodel_ce_loss() -> Trainer.Config:
    config = qwen3_debugmodel()
    config.loss = CrossEntropyLoss.Config()
    return config


def qwen3_moe_debug_ep_ce_loss() -> Trainer.Config:
    config = qwen3_moe_debug_ep()
    config.loss = CrossEntropyLoss.Config()
    return config
