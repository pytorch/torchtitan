# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified DeepSeek V3 GraphTrainer recipes."""

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.deepseek_v3 import (
    deepseek_v3_671b_dist_moe_bf16,
    deepseek_v3_671b_dist_moe_mxfp8,
)


def _dist_moe_graph_config(base_config: Trainer.Config) -> GraphTrainer.Config:
    """Convert a Dist-MoE recipe to functional-WGrad GraphTrainer execution."""
    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts

    for _, experts, _, _ in base_config.model.traverse(DistMoeRoutedExperts.Config):
        experts.inplace_wgrad_accum = False
    config = to_graph_trainer_config(base_config, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the verified 671B BF16 Dist-MoE GraphTrainer recipe."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_bf16())


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the verified 671B MXFP8 Dist-MoE GraphTrainer recipe."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_mxfp8())
