# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``h100`` integration test suite."""

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_hybridep,
)
from torchtitan.models.llama3.config_registry import (
    llama3_debugmodel,
    llama3_debugmodel_dist_gemm,
    llama3_debugmodel_float8,
)
from torchtitan.trainer import Trainer


def llama3_debugmodel_tp2_asynctp_compile() -> Trainer.Config:
    config = llama3_debugmodel()
    config.compile.enable = True
    config.parallelism.tensor_parallel_degree = 2
    config.compile.enable_async_tensor_parallel = True
    return config


def llama3_debugmodel_dist_gemm_tp2() -> Trainer.Config:
    config = llama3_debugmodel_dist_gemm()
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp_symm_mem() -> Trainer.Config:
    config = llama3_debugmodel()
    config.parallelism.enable_fsdp_symm_mem = True
    return config


def llama3_debugmodel_float8_fsdp2_tp2_pp2_asynctp_compile() -> Trainer.Config:
    config = llama3_debugmodel_float8()
    config.compile.enable = True
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.compile.enable_async_tensor_parallel = True
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_float8_hsdp2x2_cp2_compile() -> Trainer.Config:
    config = llama3_debugmodel_float8()
    config.compile.enable = True
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config


def deepseek_v3_debugmodel_minimal_async_ep_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_minimal_async_ep,
    )

    config = deepseek_v3_debugmodel_minimal_async_ep()
    config.compile.enable = False
    # TODO: Drop this once the H100 suite is migrated to the spmd_types backend.
    config.parallelism.spmd_backend = "spmd_types"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 8
    config.activation_checkpoint = FullAC.Config()
    return config


def deepseek_v3_debugmodel_hybridep_fsdp4_ep2_compile() -> Trainer.Config:
    config = deepseek_v3_debugmodel_hybridep()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.compile.enable = True
    config.compile.components = ["model", "loss"]
    return config
