# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Comparable multi-turn RL runs for measuring inter-generator routing."""

from dataclasses import replace

from torchtitan.rl.controller import Controller
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import LeastLoadedRoutingStrategy
from torchtitan.rl.examples.alphabet_sort.config_registry import rl_grpo_qwen3_0_6b_flex
from torchtitan.rl.observability.vllm import VllmOtelStatLogger


def sticky() -> Controller.Config:
    """Run multi-turn alphabet sort on two generators with the default router."""
    config = rl_grpo_qwen3_0_6b_flex()
    config.num_generators = 2
    config.generator.parallelism = replace(
        config.generator.parallelism, tensor_parallel_degree=1
    )
    config.generator.vllm_stat_logger = VllmOtelStatLogger.Config()
    config.metrics.enable_wandb = False
    config.async_loop.validation.num_samples = 0
    config.rollouter.train_dataset = replace(
        config.rollouter.train_dataset, max_turns=4, max_names_per_turn=8
    )
    return config


def least_loaded() -> Controller.Config:
    """Use the same workload with explicit least-loaded routing as baseline."""
    config = sticky()
    config.generator_router = InterGeneratorRouter.Config(
        strategy=LeastLoadedRoutingStrategy.Config()
    )
    return config
