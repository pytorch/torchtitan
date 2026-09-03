# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Hunks in this file are copied from upstream open PR 4322/4449/4450 (fegin's CP stack) to unblock running;
# pending rebase and reconcile.

"""Configurations for the ``h100`` integration test suite."""

from torchtitan.distributed.activation_checkpoint import FullAC

from torchtitan.models.common.cp_attention import AllGatherCPFlexAttention
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_hybridep,
)
from torchtitan.models.llama3.config_registry import (
    llama3_debugmodel,
    llama3_debugmodel_dist_gemm,
    llama3_debugmodel_float8,
)
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.trainer import Trainer
from torchtitan.transforms import apply_transforms, ContextParallelTransform


def llama3_debugmodel_tp2_asynctp_compile() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.compile.enable = True
    config.parallelism.tensor_parallel_degree = 2
    config.compile.enable_async_tensor_parallel = True
    return config


def llama3_debugmodel_dist_gemm_tp2() -> Trainer.Config:
    config = llama3_debugmodel_dist_gemm(seq_len=2048)
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp_symm_mem() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.enable_fsdp_symm_mem = True
    return config


def llama3_debugmodel_float8_fsdp2_tp2_pp2_asynctp_compile() -> Trainer.Config:
    config = llama3_debugmodel_float8(seq_len=2048)
    config.compile.enable = True
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.compile.enable_async_tensor_parallel = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_float8_hsdp2x2_cp2_compile() -> Trainer.Config:
    config = llama3_debugmodel_float8(seq_len=2048)
    config.compile.enable = True
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.context_parallel_degree = 2
    return apply_transforms(
        config,
        [ContextParallelTransform.Config(kernel=AllGatherCPFlexAttention)],
    )


def deepseek_v3_debugmodel_minimal_async_ep_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_minimal_async_ep,
    )

    config = deepseek_v3_debugmodel_minimal_async_ep(seq_len=2048)
    config.compile.enable = False
    # TODO: Drop this once the H100 suite is migrated to the spmd_types backend.
    config.parallelism.spmd_backend = "spmd_types"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 8
    config.activation_checkpoint = FullAC.Config()
    return apply_transforms(
        config,
        [ContextParallelTransform.Config(kernel=AllGatherCPFlexAttention)],
    )


def deepseek_v3_debugmodel_hybridep_fsdp4_ep2_compile() -> Trainer.Config:
    config = deepseek_v3_debugmodel_hybridep(seq_len=2048)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.compile.enable = True
    config.compile.components = ["model", "loss"]
    return config


def qwen3_moe_deepep_fsdp4_ep4() -> Trainer.Config:
    from torchtitan.models.qwen3.config_registry import qwen3_moe_deepep

    config = qwen3_moe_deepep(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 4
    return config


def deepseek_v3_debugmodel_minimal_async_ep_fsdp2_tp2_cp2_ep8_sdc_replay() -> (
    Trainer.Config
):
    config = deepseek_v3_debugmodel_minimal_async_ep_fsdp2_tp2_cp2_ep8()
    config.debug.deterministic = True
    config.debug.seed = 42
    config.sdc_replayer = SDCReplayer.Config()
    return config
