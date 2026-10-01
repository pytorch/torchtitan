# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Qwen3 GraphTrainer recipes."""

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.models.qwen3 import qwen3_14b


def graph_trainer_qwen3_14b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        qwen3_14b(seq_len=4096), GraphTrainerQwen3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config
