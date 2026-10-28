# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""h100 integration-test suite: run configurations and their test definitions.

Each configuration function is selected by module and name in the launched
training process, so it must stay importable from this module.
"""

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

from tests.integration_tests import IntegrationTestDefinition


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
    from torchtitan.models.common.attention import MLAAttention
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.cp_mla import AllGatherKVMLAFlexAttention
    from torchtitan.models.common.attention.kda import InnerKDA

    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    MLAAttention: AllGatherKVMLAFlexAttention,
                    InnerKDA: ContextParallelInnerKDA,
                }
            ),
        ],
    )


def kimi_k3_debugmodel_mm_ulysses_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.models.common.attention import MLAAttention
    from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.common.attention.cp_mla import UlyssesMLAAttention
    from torchtitan.models.common.attention.kda import InnerKDA

    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    MLAAttention: UlyssesMLAAttention,
                    InnerKDA: ContextParallelInnerKDA,
                }
            ),
        ],
    )


def qwen35_debugmodel_mm_allgather_kv_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        AllGatherKVFlexInnerAttention,
    )
    from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
    from torchtitan.models.common.attention.gdn import InnerGDN

    from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel

    config = qwen35_debugmodel(attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
    config.training.disable_accelerator_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: AllGatherKVFlexInnerAttention,
                    InnerGDN: ContextParallelInnerGDN,
                }
            ),
        ],
    )


def qwen35_debugmodel_mm_ulysses_cp2() -> Trainer.Config:
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        UlyssesFlexInnerAttention,
    )
    from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
    from torchtitan.models.common.attention.gdn import InnerGDN

    from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel

    config = qwen35_debugmodel(attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    config.training.disable_accelerator_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: UlyssesFlexInnerAttention,
                    InnerGDN: ContextParallelInnerGDN,
                }
            ),
        ],
    )


def build_h100_tests_list() -> list[IntegrationTestDefinition]:
    """
    Build the list of integration tests that need H100-class hardware.

    Each entry names one configuration per run.
    """
    return [
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp_symm_mem],
            test_descr="FSDP symmetric memory",
            test_name="fsdp_symm_mem",
            ngpu=2,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_hybridep_fsdp4_ep2],
            test_descr="DeepSeek V3 FSDP+HybridEP",
            test_name="deepseek_v3_fsdp+hybridep",
            ngpu=4,
            # deep_ep/NVSHMEM is CUDA-only, so skip on ROCm.
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_dist_gemm_tp2],
            test_descr="Dist GEMM: fuse the TP collectives into the attention "
            "and FFN projections (FSDP2 + TP2)",
            test_name="dist_gemm",
            ngpu=4,
            # symmetric memory is CUDA-only.
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[qwen3_moe_deepep_fsdp4_ep4],
            test_descr="Qwen3 FSDP+DeepEP",
            test_name="qwen3_fsdp+deepep",
            ngpu=4,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[qwen35_debugmodel_moe_lora],
            test_descr="Qwen3.5 MoE LoRA",
            test_name="qwen3_5_moe_lora",
            ngpu=8,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[kimi_k3_debugmodel_mm_allgather_kv_cp2],
            test_descr="Kimi K3 multimodal K/V all-gather context parallelism",
            test_name="kimi_k3_mm_allgather_kv_cp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[kimi_k3_debugmodel_mm_ulysses_cp2],
            test_descr="Kimi K3 multimodal Ulysses context parallelism",
            test_name="kimi_k3_mm_ulysses_cp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[qwen35_debugmodel_mm_allgather_kv_cp2],
            test_descr="Qwen3.5 multimodal K/V all-gather context parallelism",
            test_name="qwen3_5_mm_allgather_kv_cp",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[qwen35_debugmodel_mm_ulysses_cp2],
            test_descr="Qwen3.5 multimodal Ulysses context parallelism",
            test_name="qwen3_5_mm_ulysses_cp",
            ngpu=2,
        ),
    ]
