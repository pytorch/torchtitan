# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""models integration-test suite: run configurations and their test definitions.

Each configuration function is selected by module and name in the launched
training process, so it must stay importable from this module.
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
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
from torchtitan_recipes.tests import _set_spmd_typechecking
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

from tests.integration_tests import IntegrationTestDefinition


def _configure_fsdp_numerics(config: Trainer.Config) -> Trainer.Config:
    """Use a logical-world-eight FSDP topology for numerical comparison.

    The expert-parallel degree is part of the model's build-time sharding, so
    ``config`` keeps the one it was built with.
    """
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.context_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
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
    config = deepseek_v3_debugmodel_mtp(
        seq_len=2048, parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.override.imports = [
        "torchtitan_recipes.overrides.helion_rope.helion_complex_rope",
    ]
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_mtp_cp2(
    *, parallelism: ParallelismConfig | None = None
) -> Trainer.Config:
    config = deepseek_v3_debugmodel_mtp(seq_len=512, parallelism=parallelism)
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
    config = deepseek_v3_debugmodel_mtp_cp2(
        parallelism=ParallelismConfig(
            expert_parallel_degree=2, enable_sequence_parallel=True
        )
    )
    config.parallelism.tensor_parallel_degree = 2
    return config


def deepseek_v3_debugmodel_fsdp8_ep8() -> Trainer.Config:
    return _configure_fsdp_numerics(
        deepseek_v3_debugmodel(
            seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=8)
        )
    )


def deepseek_v3_debugmodel_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    config = deepseek_v3_debugmodel(
        seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=8)
    )
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


def deepseek_v3_debugmodel_fsdp2_cp2_pp2_ep4() -> Trainer.Config:
    config = deepseek_v3_debugmodel(
        seq_len=2048, parallelism=ParallelismConfig(expert_parallel_degree=4)
    )
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
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
    config = deepseek_v3_debugmodel(
        seq_len=2048, parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v4_debugmodel_fsdp2_tp2_ep2() -> Trainer.Config:
    from torchtitan_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel(
        seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_fused_mla_swiglu_fsdp4_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel(
        seq_len=2048, parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    _set_spmd_typechecking(config, typechecking=True)
    config.override.imports = [
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    config.parallelism.data_parallel_shard_degree = 4
    config.training.disable_cuda_graphs = True
    return config


def qwen3_debugmodel_moe_param_groups_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    config = qwen3_debugmodel_moe_param_groups(
        seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=8)
    )
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


def qwen3_debugmodel_moe_param_groups_seed() -> Trainer.Config:
    config = qwen3_debugmodel_moe_param_groups()
    config.checkpointer = CheckpointManager.Config(export_dtype="float16")
    return config


def qwen3_debugmodel_fsdp2_tp2_cp2(
    *, parallelism: ParallelismConfig | None = None
) -> Trainer.Config:
    config = qwen3_debugmodel(seq_len=2048, parallelism=parallelism)
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
    config = qwen3_debugmodel_fsdp2_tp2_cp2(
        parallelism=ParallelismConfig(enable_sequence_parallel=False)
    )
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
    config = gpt_oss_debugmodel(
        seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=4)
    )
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.dataloader.max_num_documents = None
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac() -> Trainer.Config:
    config = gpt_oss_debugmodel_flex(
        seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=4)
    )
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
    config = gpt_oss_debugmodel(
        seq_len=2048, parallelism=ParallelismConfig(expert_parallel_degree=4)
    )
    _set_spmd_typechecking(config, typechecking=False)
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.training.num_tokens_per_train_step = 131072
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
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

    config = kimi_k2_5_debugmodel(
        seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.data_parallel_shard_degree = 2
    # Four microbatches match the four virtual pipeline stages.
    config.parallelism.num_pp_microbatches = 4
    config.training.steps = 1
    config.training.disable_cuda_graphs = True
    set_rank_conditional_image_presence(config)
    return config


def kimi_k2_5_debugmodel_muon_fsdp8_ep8() -> Trainer.Config:
    from torchtitan_recipes.tests.models.kimi_k2_7 import kimi_k2_5_debugmodel

    config = kimi_k2_5_debugmodel(
        seq_len=512, parallelism=ParallelismConfig(expert_parallel_degree=8)
    )
    config.parallelism.data_parallel_shard_degree = 8
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


def build_model_tests_list() -> list[IntegrationTestDefinition]:
    """
    Build the list of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will
    broadly cover all the supported model parallelism patterns on all the supported
    models.
    """
    return [
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp2_tp2_cp2],
            test_descr="Llama 3 FSDP+TP+CP",
            test_name="llama3_fsdp+tp+cp",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/llama3.txt"
            ),
            loss_compare_seed_config=llama3_debugmodel,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_region_ac_fsdp2_tp2_cp2],
            test_descr="Llama 3 FSDP+TP+CP+RegionAC",
            test_name="llama3_fsdp+tp+cp+region_ac",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/llama3.txt"
            ),
            loss_compare_seed_config=llama3_debugmodel,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp2_tp2_pp2],
            test_descr="Llama 3 FSDP+TP+PP",
            test_name="llama3_fsdp+tp+pp",
            ngpu=8,
            golden_numerics_path="tests/assets/losses/real_pg/{gpu_arch}/llama3_pp.txt",
            use_real_pg=True,
        ),
        # Integration Test Cases for DeepSeek V3
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_mtp_fsdp4_ep2],
            test_descr="DeepSeek V3 MTP FSDP+EP",
            test_name="deepseek_v3_mtp_fsdp+ep",
            ngpu=4,
            # The Helion fused RoPE kernels are CUDA-only and tuned for NVIDIA
            # H100/GB200; skip on ROCm where they are unvalidated.
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_mtp_tp2_cp2],
            test_descr="DeepSeek V3 MTP TP+CP with SP",
            test_name="deepseek_v3_mtp_tp+cp",
            ngpu=4,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_fsdp8_ep8],
            test_descr="DeepSeek V3 FSDP+EP",
            test_name="deepseek_v3_fsdp+ep",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/deepseek_v3.txt"
            ),
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_fsdp2_cp2_pp2_ep4],
            test_descr="DeepSeek V3 FSDP+CP+PP+EP",
            test_name="deepseek_v3_fsdp+cp+pp+ep",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/real_pg/{gpu_arch}/deepseek_v3_cp_pp.txt"
            ),
            loss_compare_seed_config=deepseek_v3_debugmodel,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_hsdp2x2_ep2],
            test_descr="DeepSeek V3 HSDP+EP",
            test_name="deepseek_v3_hsdp+ep",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_fused_mla_swiglu_fsdp4_ep2],
            test_descr="DeepSeek V3 fused MLA+SwiGLU FSDP+EP",
            test_name="deepseek_v3_fused_mla_swiglu_fsdp+ep",
            ngpu=4,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v4_debugmodel_fsdp2_tp2_ep2],
            test_descr="DeepSeek V4 FSDP+TP+EP",
            test_name="deepseek_v4_fsdp+tp+ep",
            ngpu=4,
            # Sparse attention / indexer kernels are CUDA-only and unvalidated
            # on ROCm.
            skip_rocm_test=True,
            # Runs on a real PG. Under Fake PG this config's sequence-parallel
            # collectives return activations that alias their inputs, which
            # corrupts a saved-for-backward tensor and blows up grad_norm at
            # step 1. The same config trains cleanly on a real 4-GPU PG
            # (grad_norm ~3.8), so keep it on a real PG until the Fake PG
            # collective aliasing under spmd_types is fixed.
            use_real_pg=True,
        ),
        # Integration Test Cases for Qwen3 dense and MoE model
        IntegrationTestDefinition(
            configs=[qwen3_debugmodel_moe_param_groups_fsdp2_tp2_cp2_ep8],
            test_descr="Qwen3 MoE FSDP+TP+CP+EP (param groups)",
            test_name="qwen3_moe_fsdp+tp+cp+ep_param_groups",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/qwen3.txt"
            ),
            loss_compare_seed_config=qwen3_debugmodel_moe_param_groups_seed,
        ),
        IntegrationTestDefinition(
            configs=[
                qwen3_debugmodel_fsdp2_tp2_cp2_no_sp,
                qwen3_debugmodel_fsdp2_tp2_cp2,
            ],
            test_descr="Qwen3 FSDP+TP+CP (SP disabled)",
            test_name="qwen3_fsdp+tp+cp_no_sp",
            ngpu=8,
        ),
        # Integration Test Cases for Qwen3.5
        IntegrationTestDefinition(
            configs=[qwen35_debugmodel_moe_fsdp2_tp2_pp2_ep4],
            test_descr="Qwen3.5 MoE FSDP+TP+EP+PP",
            test_name="qwen3_5_moe_fsdp+tp+ep+pp",
            ngpu=8,
            use_real_pg=True,
            # short_conv's CuTe/CUTLASS kernel (attn_gym) is CUDA-only.
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[qwen35_debugmodel_moe_fsdp4_tp2_ep4],
            test_descr="Qwen3.5 MoE FSDP+TP+EP",
            test_name="qwen3_5_moe_fsdp+tp+ep",
            ngpu=8,
            # NOTE: This topology is not bitwise deterministic with Real PG on
            # A10G, so this case provides end-to-end coverage without a golden.
            # short_conv's CuTe/CUTLASS kernel (attn_gym) is CUDA-only.
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[qwen35_debugmodel_varlen_attn_fsdp2_tp2_sac],
            test_descr="Qwen3.5 FSDP+TP+VARLEN_ATTN + selective AC",
            test_name="qwen3_5_fsdp+tp+varlen_attn+per_op_sac",
            ngpu=4,
            skip_rocm_test=True,
            use_real_pg=True,
        ),
        # Integration Test Cases for gpt-oss
        IntegrationTestDefinition(
            configs=[gpt_oss_debugmodel_fsdp4_tp2_ep4],
            test_descr="GPT-OSS FSDP+TP+EP",
            test_name="gpt_oss_fsdp+tp+ep",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/gpt_oss.txt"
            ),
        ),
        IntegrationTestDefinition(
            configs=[gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac],
            test_descr="GPT-OSS PP+FSDP+CP+EP+selective AC",
            test_name="gpt_oss_pp+fsdp+cp+ep+sacop",
            ngpu=8,
            golden_numerics_path="tests/assets/losses/real_pg/{gpu_arch}/gpt_oss_pp.txt",
            loss_compare_seed_config=gpt_oss_debugmodel_flex,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[gpt_oss_debugmodel_fsdp4_pp2_ep4_sac],
            test_descr="GPT-OSS PP+FSDP+EP+selective AC with VarlenInnerAttention",
            test_name="gpt_oss_pp+fsdp+ep+sacop",
            ngpu=8,
            use_real_pg=True,
        ),
        # Integration Test Cases for Kimi K2.7
        IntegrationTestDefinition(
            configs=[kimi_k2_5_debugmodel_muon_fsdp2_pp2_ep2],
            test_descr="Kimi K2.7 DistMuon PP+FSDP+EP",
            test_name="kimi_k2_5_muon_pp+fsdp+ep",
            ngpu=4,
            timeout=600,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[kimi_k2_5_debugmodel_muon_fsdp8_ep8],
            test_descr="Kimi K2.5 DistMuon FSDP+EP",
            test_name="kimi_k2_5_muon_fsdp+ep",
            ngpu=8,
        ),
        # Integration Test Cases for Muse Glimmer
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_fsdp8],
            test_descr="Muse Glimmer text FSDP",
            test_name="muse_glimmer_text_fsdp",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/muse_glimmer.txt"
            ),
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_mm_fsdp2_tp2],
            test_descr="Muse Glimmer multimodal FSDP+TP+SP",
            test_name="muse_glimmer_mm_fsdp+tp+sp",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_mm_tp2_cp2_pp2],
            test_descr="Muse Glimmer multimodal TP+CP+PP+SP",
            test_name="muse_glimmer_mm_tp+cp+pp+sp",
            ngpu=8,
            use_real_pg=True,
        ),
    ]
