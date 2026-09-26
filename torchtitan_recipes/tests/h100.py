# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``h100`` integration test suite."""

from torchtitan.models.llama3.config_registry import (
    llama3_debugmodel,
    llama3_debugmodel_dist_gemm,
)
from torchtitan.models.qwen3_5.config_registry import (
    qwen35_debugmodel_moe_float8_lora as _qwen35_debugmodel_moe_float8_lora,
)
from torchtitan.trainer import Trainer


def llama3_debugmodel_dist_gemm_tp2() -> Trainer.Config:
    config = llama3_debugmodel_dist_gemm(seq_len=2048)
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp_symm_mem() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.fsdp_symm_mem_scope = "all"
    return config


def qwen3_moe_deepep_fsdp4_ep4() -> Trainer.Config:
    from torchtitan.models.qwen3.config_registry import qwen3_moe_deepep

    config = qwen3_moe_deepep(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 4
    return config


def qwen35_debugmodel_moe_float8_lora() -> Trainer.Config:
    return _qwen35_debugmodel_moe_float8_lora(seq_len=512)
