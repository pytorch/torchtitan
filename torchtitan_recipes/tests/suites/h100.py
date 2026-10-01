# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the H100 integration-test suite."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel_hybridep
from torchtitan_recipes.tests.models.llama3 import (
    llama3_debugmodel,
    llama3_debugmodel_dist_gemm,
)


def llama3_debugmodel_dist_gemm_tp2() -> Trainer.Config:
    config = llama3_debugmodel_dist_gemm(seq_len=2048)
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp_symm_mem() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.fsdp_symm_mem_scope = "all"
    return config


def deepseek_v3_debugmodel_hybridep_fsdp4_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel_hybridep(seq_len=2048)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    return config


def qwen3_moe_deepep_fsdp4_ep4() -> Trainer.Config:
    from torchtitan_recipes.tests.models.qwen3 import qwen3_moe_deepep

    config = qwen3_moe_deepep(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 4
    return config
