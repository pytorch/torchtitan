# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _use_spmd_types


def kimi_k3_debugmodel_mm_fsdp2() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    return config


def llama3_debugmodel_mxfp8_fsdp2() -> Trainer.Config:
    from torchtitan.models.llama3.config_registry import llama3_debugmodel_mxfp8

    config = llama3_debugmodel_mxfp8()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2() -> Trainer.Config:
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_dist_moe_bf16,
    )

    config = deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.steps = 4
    config.checkpoint.enable = False
    return config


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2() -> Trainer.Config:
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_dist_moe_mxfp8,
    )

    config = deepseek_v3_debugmodel_dist_moe_mxfp8(seq_len=128)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.steps = 4
    config.checkpoint.enable = False
    return config
