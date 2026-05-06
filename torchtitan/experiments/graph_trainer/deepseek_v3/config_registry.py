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
from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
    PagedStashMemoryPolicy,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_ep,
    deepseek_v3_debugmodel_flex_attn,
    deepseek_v3_debugmodel_flex_attn_ep,
)
from torchtitan.trainer import Trainer

from . import model_registry


def graph_trainer_deepseek_v3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel_ep(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_hybridep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel_ep(), model_registry)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        memory_policy=PagedStashMemoryPolicy(),
    )
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def graph_trainer_deepseek_v3_debugmodel_flex_attn() -> (GraphTrainer.Config):
    config = to_graph_trainer_config(deepseek_v3_debugmodel_flex_attn(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_flex_attn_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_debugmodel_flex_attn_ep(), model_registry
    )
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


# CrossEntropyLoss baseline for graph_trainer numerics tests. graph_trainer
# doesn't yet support ChunkedCELoss, so to_graph_trainer_config swaps it for
# CrossEntropyLoss; this wrapper applies the same swap to the eager baseline
# so loss_compare runs apples-to-apples.
# TODO: Remove once graph_trainer supports ChunkedCELoss.
def deepseek_v3_debugmodel_ep_ce_loss() -> Trainer.Config:
    config = deepseek_v3_debugmodel_ep()
    config.loss = CrossEntropyLoss.Config()
    return config
