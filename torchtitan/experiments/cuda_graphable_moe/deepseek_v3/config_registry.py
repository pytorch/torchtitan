# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.cuda_graphable_moe.configs import to_paged_stash_config
from torchtitan.experiments.cuda_graphable_moe.train import PagedStashTrainer
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
)


def _apply_defaults(config: PagedStashTrainer.Config) -> None:
    config.model_spec = model_registry(
        config.model_spec.flavor,
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        memory_policy="paged_stash",
    )


def paged_stash_deepseek_v3_debugmodel() -> PagedStashTrainer.Config:
    config = to_paged_stash_config(deepseek_v3_debugmodel())
    _apply_defaults(config)
    return config


def paged_stash_deepseek_v3_671b() -> PagedStashTrainer.Config:
    config = to_paged_stash_config(deepseek_v3_671b())
    _apply_defaults(config)
    return config
