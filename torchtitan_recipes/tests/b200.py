# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from torchtitan.components.optimizer import default_adamw
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _set_spmd_typechecking


def kimi_k3_debugmodel_mm() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    # DistMuon rejects TP-produced _StridedShard storage, so the TP coverage
    # keeps AdamW; kimi_k3_debugmodel_mm_muon covers the default optimizer.
    config.optimizer = default_adamw(lr=8e-4)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.expert_parallel_degree = 2
    return config


def kimi_k3_debugmodel_mm_muon() -> Trainer.Config:
    """Per-head DistMuon with FSDP and EP."""
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    return config


def llama3_debugmodel_mxfp8_fsdp2() -> Trainer.Config:
    from torchtitan.models.llama3.config_registry import llama3_debugmodel_mxfp8

    config = llama3_debugmodel_mxfp8()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def llama3_debugmodel_nvfp4_fsdp2() -> Trainer.Config:
    from torchtitan.config import CompileConfig
    from torchtitan.models.llama3.config_registry import llama3_debugmodel_nvfp4

    config = llama3_debugmodel_nvfp4(seq_len=2048)
    config.compile = CompileConfig(components=["model"])
    config.parallelism.data_parallel_shard_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def kimi_k3_debugmodel_mm_allgather_kv_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.distributed.context_parallel import HeadTailLoadBalancer
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
    from torchtitan.models.kimi_k3.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.kimi_k3.kda import InnerKDA

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = HeadTailLoadBalancer.Config()
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention={
                    FlexInnerAttention.Config: KVAllGatherCPFlexInnerAttention,
                    InnerKDA.Config: ContextParallelInnerKDA,
                }
            ),
        ],
    )


def kimi_k3_debugmodel_mm_ulysses_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.cp_attention import UlyssesCPFlexInnerAttention
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
    from torchtitan.models.kimi_k3.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.kimi_k3.kda import InnerKDA

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention={
                    FlexInnerAttention.Config: UlyssesCPFlexInnerAttention,
                    InnerKDA.Config: ContextParallelInnerKDA,
                }
            ),
        ],
    )
