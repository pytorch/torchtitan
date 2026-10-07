# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the H100 integration-test suite."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _set_spmd_typechecking
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel_hybridep
from torchtitan_recipes.tests.models.llama3 import (
    llama3_debugmodel,
    llama3_debugmodel_dist_gemm,
)
from torchtitan_recipes.tests.models.qwen3_5 import (
    qwen35_debugmodel_moe_lora as _qwen35_debugmodel_moe_lora,
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


def qwen35_debugmodel_moe_lora() -> Trainer.Config:
    return _qwen35_debugmodel_moe_lora(seq_len=512)


def kimi_k3_debugmodel_mm_allgather_kv_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        KVAllGatherCPFlexInnerAttention,
    )
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.kda import InnerKDA

    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                    InnerKDA: ContextParallelInnerKDA,
                }
            ),
        ],
    )


def kimi_k3_debugmodel_mm_ulysses_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        UlyssesCPFlexInnerAttention,
    )
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.kda import InnerKDA

    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: UlyssesCPFlexInnerAttention,
                    InnerKDA: ContextParallelInnerKDA,
                }
            ),
        ],
    )
