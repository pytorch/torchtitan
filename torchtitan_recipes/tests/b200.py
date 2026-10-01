# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from torchtitan.components.optimizer import AdamW, OptimizersContainer
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _set_spmd_typechecking
from torchtitan_recipes.tests.multimodal import set_rank_conditional_image_presence


def kimi_k3_debugmodel_mm() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    # DistMuon rejects TP-produced _StridedShard storage, so the TP coverage
    # keeps AdamW; kimi_k3_debugmodel_mm_muon covers the default optimizer.
    config.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.expert_parallel_degree = 2
    set_rank_conditional_image_presence(config)
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


def kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4() -> Trainer.Config:
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
    )
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
    from torchtitan.models.kimi_k3.model import KimiK3Model

    config = kimi_k3_debugmodel()
    # Type checking stays off under pipeline parallelism, as the other pipeline
    # recipes have it.
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    # Four stages per rank (the default is two), the shape where a hop can carry
    # no new block and a block has three later readers on its rank: core's split
    # for that many stages, with the model's end modules pinned.
    parallelism = config.parallelism
    split = _generate_llm_fqn_per_model_part(
        4 * parallelism.pipeline_parallel_degree,
        len(config.model.layers),
        parallelism.pipeline_parallel_first_stage_less_layers,
        parallelism.pipeline_parallel_last_stage_less_layers,
    )
    split[0][:0] = KimiK3Model.pipeline_first_stage_module_fqns
    split[-1].extend(KimiK3Model.pipeline_last_stage_module_fqns)
    parallelism.pipeline_parallel_module_fqns_per_model_part = split
    # DistMuon does not support tensor parallelism yet (#3353), so this cell
    # keeps AdamW the way kimi_k3_debugmodel_mm does.
    config.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    return config


def _configure_dist_moe_fsdp2_ep2(config: Trainer.Config) -> Trainer.Config:
    """Apply the common two-GPU Dist-MoE integration-test topology."""
    runtime = config.dist_moe
    assert isinstance(runtime, DistMoeRuntime.Config)
    runtime.scratch_capacity_factor = 2.0
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 128
    config.training.max_context_length = 128
    config.training.steps = 4
    config.checkpointer = None
    return config


def deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2() -> Trainer.Config:
    """Exercise BF16 Dist-MoE in-place WGRAD accumulation with FSDP and EP."""
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_dist_moe_bf16,
    )

    return _configure_dist_moe_fsdp2_ep2(
        deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    )


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2() -> Trainer.Config:
    """Exercise MXFP8 Dist-MoE in-place WGRAD accumulation with FSDP and EP."""
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_dist_moe_mxfp8,
    )

    return _configure_dist_moe_fsdp2_ep2(
        deepseek_v3_debugmodel_dist_moe_mxfp8(seq_len=128)
    )


def graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2() -> Trainer.Config:
    """Exercise BF16 Dist-MoE in non-pipeline GraphTrainer."""
    from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
        graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16,
    )

    return _configure_dist_moe_fsdp2_ep2(
        graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16()
    )


def graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2() -> Trainer.Config:
    """Exercise MXFP8 Dist-MoE in non-pipeline GraphTrainer."""
    from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
        graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8,
    )

    return _configure_dist_moe_fsdp2_ep2(
        graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8()
    )


def graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2() -> (
    Trainer.Config
):
    """Exercise slot-specialized Dist-MoE graphs in GraphPP."""
    config = graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    return config


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2() -> Trainer.Config:
    """Exercise schedule-derived Dist-MoE slots in eager pipeline parallelism."""
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_dist_moe_mxfp8,
    )

    config = _configure_dist_moe_fsdp2_ep2(
        deepseek_v3_debugmodel_dist_moe_mxfp8(seq_len=128)
    )
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    return config


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2_fp32_reduce() -> (
    Trainer.Config
):
    """Exercise eager PP with BF16 Dist-MoE WGrad and FP32 reduction."""
    config = deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2()
    config.training.mixed_precision_reduce = "float32"
    return config


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_vmm() -> Trainer.Config:
    """Exercise host-backed VMM scratch preallocation with MXFP8 Dist-MoE."""
    config = deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2()
    runtime = config.dist_moe
    assert isinstance(runtime, DistMoeRuntime.Config)
    runtime.scratch_capacity_factor = 1.0
    runtime.vmm_capacity_factor = 4.0
    return config
