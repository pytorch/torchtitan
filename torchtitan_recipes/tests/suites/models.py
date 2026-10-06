# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the model integration-test suite."""

from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.distributed.activation_checkpoint import RegionAC, SelectiveAC
from torchtitan.distributed.context_parallel import PTRRFlexAttentionCPLoadBalancer
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    KVAllGatherCPSlidingWindowFlexInnerAttention,
)
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_mtp,
)
from torchtitan_recipes.tests.models.gpt_oss import (
    gpt_oss_debugmodel,
    gpt_oss_debugmodel_flex,
)
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel
from torchtitan_recipes.tests.models.multimodal import (
    set_rank_conditional_image_presence,
)
from torchtitan_recipes.tests.models.qwen3 import (
    qwen3_debugmodel,
    qwen3_debugmodel_moe_param_groups,
)

from .. import _set_spmd_typechecking


def _configure_fsdp_numerics(
    config: Trainer.Config, *, expert_parallel_degree: int = 1
) -> Trainer.Config:
    """Use a logical-world-eight FSDP topology for numerical comparison."""
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.context_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = expert_parallel_degree
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                }
            )
        ],
    )


def llama3_debugmodel_region_ac_fsdp2_tp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_tp2_cp2()
    config.activation_checkpoint = RegionAC.Config(
        save_regions=[
            "attention.qkv_linear.wqkv.linear",
            "attention.inner_attention",
            "attention.wo.linear",
        ]
    )
    return config


def llama3_debugmodel_fsdp2_tp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_mtp_fsdp4_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel_mtp(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.override.imports = [
        "torchtitan_recipes.overrides.helion_rope.helion_complex_rope",
    ]
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_mtp_cp2() -> Trainer.Config:
    config = deepseek_v3_debugmodel_mtp(seq_len=512)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.context_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                }
            )
        ],
    )


def deepseek_v3_debugmodel_mtp_tp2_cp2() -> Trainer.Config:
    config = deepseek_v3_debugmodel_mtp_cp2()
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    return config


def deepseek_v3_debugmodel_fsdp8_ep8() -> Trainer.Config:
    return _configure_fsdp_numerics(
        deepseek_v3_debugmodel(seq_len=512),
        expert_parallel_degree=8,
    )


def deepseek_v3_debugmodel_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 8
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                }
            )
        ],
    )


def deepseek_v3_debugmodel_fsdp2_cp2_pp2_ep4() -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                }
            )
        ],
    )


def deepseek_v3_debugmodel_hsdp2x2_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v4_debugmodel_fsdp2_tp2_ep2() -> Trainer.Config:
    from torchtitan_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel(seq_len=512)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_fused_mla_swiglu_fsdp4_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.override.imports = [
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def qwen3_debugmodel_moe_param_groups_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    config = qwen3_debugmodel_moe_param_groups(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 8
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                }
            )
        ],
    )


def qwen3_debugmodel_moe_param_groups_seed() -> Trainer.Config:
    config = qwen3_debugmodel_moe_param_groups()
    config.checkpointer = CheckpointManager.Config(export_dtype="float16")
    return config


def qwen3_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    config = qwen3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                }
            )
        ],
    )


def qwen3_debugmodel_fsdp2_tp2_cp2_no_sp() -> Trainer.Config:
    config = qwen3_debugmodel_fsdp2_tp2_cp2()
    config.parallelism.enable_sequence_parallel = False
    return config


def qwen35_debugmodel_moe_fsdp2_tp2_pp2_ep4() -> Trainer.Config:
    from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel_moe

    config = qwen35_debugmodel_moe(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    _set_spmd_typechecking(config, typechecking=False)
    config.training.disable_cuda_graphs = True
    set_rank_conditional_image_presence(config)
    return config


def qwen35_debugmodel_moe_fsdp4_tp2_ep4() -> Trainer.Config:
    from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel_moe

    config = qwen35_debugmodel_moe(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.pipeline_parallel_degree = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        config.training.max_context_length
    )
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    set_rank_conditional_image_presence(config)
    return config


def qwen35_debugmodel_varlen_attn_fsdp2_tp2_sac() -> Trainer.Config:
    from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel_varlen_attn

    config = qwen35_debugmodel_varlen_attn(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.activation_checkpoint = SelectiveAC.Config()
    _set_spmd_typechecking(config, typechecking=False)
    config.training.disable_cuda_graphs = True
    set_rank_conditional_image_presence(config)
    return config


def gpt_oss_debugmodel_fsdp4_tp2_ep4() -> Trainer.Config:
    config = gpt_oss_debugmodel(seq_len=512)
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.dataloader.max_num_documents = None
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac() -> Trainer.Config:
    config = gpt_oss_debugmodel_flex(seq_len=512)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = (
        PTRRFlexAttentionCPLoadBalancer.Config()
    )
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.parallelism.expert_parallel_degree = 4
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.disable_cuda_graphs = True
    config.training.steps = 10
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                    SlidingWindowFlexInnerAttention: (
                        KVAllGatherCPSlidingWindowFlexInnerAttention
                    ),
                }
            )
        ],
    )


def gpt_oss_debugmodel_fsdp4_pp2_ep4_sac() -> Trainer.Config:
    config = gpt_oss_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.training.num_tokens_per_train_step = 131072
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.expert_parallel_degree = 4
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.disable_cuda_graphs = True
    return config


def kimi_k2_5_debugmodel_muon_fsdp2_pp2_ep2() -> Trainer.Config:
    """One four-GPU smoke path covering PP=2, FSDP=2, and EP=2.

    Each PP stage consumes its local subset of the global DistMuon
    compute-sharding map. DistMuon rejects tensor parallel: it produces
    _StridedShard storage.
    """
    from torchtitan_recipes.tests.models.kimi_k2_7 import kimi_k2_5_debugmodel

    config = kimi_k2_5_debugmodel(seq_len=512)
    config = replace(
        config,
        parallelism=replace(config.parallelism, expert_parallel_degree=2),
    )
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    # Four microbatches match the four virtual pipeline stages.
    config.parallelism.num_pp_microbatches = 4
    config.training.steps = 1
    config.training.disable_cuda_graphs = True
    set_rank_conditional_image_presence(config)
    return config


def kimi_k2_5_debugmodel_muon_fsdp8_ep8() -> Trainer.Config:
    from torchtitan_recipes.tests.models.kimi_k2_7 import kimi_k2_5_debugmodel

    config = kimi_k2_5_debugmodel(seq_len=512)
    config = replace(
        config,
        parallelism=replace(config.parallelism, expert_parallel_degree=8),
    )
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.expert_parallel_degree = 8
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    set_rank_conditional_image_presence(config)
    return config


def kimi_k2_5_debugmodel_seed_checkpoint() -> Trainer.Config:
    """Use the same Kimi model with an optimizer safe for unsharded setup."""
    config = kimi_k2_5_debugmodel_muon_fsdp8_ep8()
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", fused=False)]
    )
    return config


def muse_glimmer_debugmodel_fsdp8() -> Trainer.Config:
    from torchtitan_recipes.tests.models.muse_glimmer import muse_glimmer_debugmodel

    return _configure_fsdp_numerics(muse_glimmer_debugmodel(seq_len=512))


def muse_glimmer_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    from torchtitan_recipes.tests.models.muse_glimmer import muse_glimmer_debugmodel

    config = muse_glimmer_debugmodel(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        config.training.max_context_length
    )
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                    SlidingWindowFlexInnerAttention: (
                        KVAllGatherCPSlidingWindowFlexInnerAttention
                    ),
                }
            )
        ],
    )


def muse_glimmer_debugmodel_mm_fsdp2_tp2() -> Trainer.Config:
    from torchtitan_recipes.tests.models.muse_glimmer import muse_glimmer_debugmodel_mm

    config = muse_glimmer_debugmodel_mm(seq_len=512)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    set_rank_conditional_image_presence(config)
    return config


def muse_glimmer_debugmodel_mm_tp2_cp2_pp2() -> Trainer.Config:
    from torchtitan_recipes.tests.models.muse_glimmer import muse_glimmer_debugmodel_mm

    config = muse_glimmer_debugmodel_mm()
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                    SlidingWindowFlexInnerAttention: (
                        KVAllGatherCPSlidingWindowFlexInnerAttention
                    ),
                }
            )
        ],
    )
