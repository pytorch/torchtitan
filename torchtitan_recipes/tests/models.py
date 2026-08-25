# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``models`` integration test suite."""

from torchtitan.components.optimizer import default_adamw
from torchtitan.distributed.activation_checkpoint import SelectiveAC

from torchtitan.models.common.cp_attention import (
    AllGatherCPFlexAttention,
    use_cp_kernel,
)
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_mtp,
)
from torchtitan.models.gpt_oss.config_registry import (
    gpt_oss_debugmodel,
    gpt_oss_debugmodel_flex,
)
from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan.models.qwen3.config_registry import (
    qwen3_debugmodel,
    qwen3_debugmodel_moe_param_groups,
    qwen3_debugmodel_non_fused_qkv,
)
from torchtitan.trainer import Trainer

from . import _use_spmd_types


def _configure_fake_pg_numerics(
    config: Trainer.Config, *, expert_parallel_degree: int = 1
) -> Trainer.Config:
    """Use a stable logical-world-eight topology for Fake-PG numerics."""
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.context_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = expert_parallel_degree
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel()
    use_cp_kernel(config, AllGatherCPFlexAttention)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_tp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_mtp_fsdp4_ep2_compile() -> Trainer.Config:
    config = deepseek_v3_debugmodel_mtp()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.compile.enable = True
    config.override.imports = [
        "torchtitan.overrides.helion_rope.helion_cos_sin_rope",
        "torchtitan.overrides.helion_rope.helion_complex_rope",
    ]
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_fsdp8_ep8() -> Trainer.Config:
    return _configure_fake_pg_numerics(
        deepseek_v3_debugmodel(), expert_parallel_degree=8
    )


def deepseek_v3_debugmodel_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    use_cp_kernel(config, AllGatherCPFlexAttention)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 8
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_fsdp2_tp2_pp2_ep4() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_hsdp2x2_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_fused_mla_swiglu_fsdp4_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.override.imports = [
        "torchtitan.overrides.fused_mla.fused_mla",
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
    ]
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def qwen3_debugmodel_moe_param_groups_fsdp2_tp2_ep4() -> Trainer.Config:
    config = qwen3_debugmodel_moe_param_groups()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.training.disable_cuda_graphs = True
    return config


def qwen3_debugmodel_moe_param_groups_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    config = qwen3_debugmodel_moe_param_groups()
    use_cp_kernel(config, AllGatherCPFlexAttention)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 8
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def qwen3_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    config = qwen3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    use_cp_kernel(config, AllGatherCPFlexAttention)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config


def qwen3_debugmodel_fsdp2_tp2_cp2_no_sp() -> Trainer.Config:
    config = qwen3_debugmodel_fsdp2_tp2_cp2()
    config.parallelism.enable_sequence_parallel = False
    return config


def qwen3_debugmodel_fsdp2_tp2_cp2_compile_helion_rope() -> Trainer.Config:
    config = qwen3_debugmodel()
    _use_spmd_types(config, typechecking=False)
    use_cp_kernel(config, AllGatherCPFlexAttention)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.compile.enable = True
    config.override.imports = ["torchtitan.overrides.helion_rope.helion_cos_sin_rope"]
    return config


def qwen3_debugmodel_non_fused_qkv_fsdp2_tp2_cp2() -> Trainer.Config:
    config = qwen3_debugmodel_non_fused_qkv()
    _use_spmd_types(config, typechecking=True)
    use_cp_kernel(config, AllGatherCPFlexAttention)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config


def qwen35_debugmodel_moe_fsdp2_tp2_pp2_ep4() -> Trainer.Config:
    from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel_moe

    config = qwen35_debugmodel_moe()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    _use_spmd_types(config, typechecking=False)
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_moe_fsdp4_tp2_ep4() -> Trainer.Config:
    from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel_moe

    config = qwen35_debugmodel_moe()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.pipeline_parallel_degree = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        config.training.max_context_length
    )
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_varlen_attn_fsdp2_tp2_sac() -> Trainer.Config:
    from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel_varlen_attn

    config = qwen35_debugmodel_varlen_attn()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    # First-run FLA/TileLang kernel compile and autotune exceed the default
    # 100s train timeout.
    config.comm.train_timeout_seconds = 600
    config.activation_checkpoint = SelectiveAC.Config()
    _use_spmd_types(config, typechecking=False)
    config.training.disable_cuda_graphs = True
    return config


def gpt_oss_debugmodel_fsdp4_tp2_ep4_compile() -> Trainer.Config:
    config = gpt_oss_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.compile.enable = True
    config.training.disable_cuda_graphs = True
    return config


def gpt_oss_debugmodel_fsdp4_tp2_ep4() -> Trainer.Config:
    config = gpt_oss_debugmodel()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac() -> Trainer.Config:
    config = gpt_oss_debugmodel_flex()
    _use_spmd_types(config, typechecking=False)
    use_cp_kernel(config, AllGatherCPFlexAttention)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = "ptrr"
    config.parallelism.context_parallel_ptrr_mask_key = "basic_mask"
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.parallelism.expert_parallel_degree = 4
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.disable_cuda_graphs = True
    config.training.max_context_length = 512
    config.training.steps = 10
    return config


def gpt_oss_debugmodel_fsdp4_pp2_ep4_sac() -> Trainer.Config:
    config = gpt_oss_debugmodel()
    _use_spmd_types(config, typechecking=False)
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
    from torchtitan.models.kimi_k2_7.config_registry import kimi_k2_5_debugmodel

    config = kimi_k2_5_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    # Four microbatches match the four virtual pipeline stages.
    config.parallelism.num_pp_microbatches = 4
    config.training.steps = 1
    config.training.disable_cuda_graphs = True
    return config


def kimi_k2_5_debugmodel_muon_fsdp8_ep8() -> Trainer.Config:
    from torchtitan.models.kimi_k2_7.config_registry import kimi_k2_5_debugmodel

    config = kimi_k2_5_debugmodel()
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.expert_parallel_degree = 8
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def kimi_k2_5_debugmodel_seed_checkpoint() -> Trainer.Config:
    """Use the same Kimi model with an optimizer safe for unsharded setup."""
    config = kimi_k2_5_debugmodel_muon_fsdp8_ep8()
    config.optimizer = default_adamw()
    config.optimizer.implementation = "for-loop"
    return config


def kimi_k3_debugmodel_mm_fsdp2() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def muse_glimmer_debugmodel_fsdp8() -> Trainer.Config:
    from torchtitan.models.muse_glimmer.config_registry import muse_glimmer_debugmodel

    return _configure_fake_pg_numerics(muse_glimmer_debugmodel())


def muse_glimmer_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    from torchtitan.models.muse_glimmer.config_registry import muse_glimmer_debugmodel

    config = muse_glimmer_debugmodel()
    use_cp_kernel(config, AllGatherCPFlexAttention)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        config.training.max_context_length
    )
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def muse_glimmer_debugmodel_mm_fsdp2_tp2() -> Trainer.Config:
    from torchtitan.models.muse_glimmer.config_registry import (
        muse_glimmer_debugmodel_mm,
    )

    config = muse_glimmer_debugmodel_mm()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config
