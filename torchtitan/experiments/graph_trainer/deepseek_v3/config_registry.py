# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.data import ConcatThenSplitPackingConfig
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.quantization import (
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_16b_dist_moe_bf16,
    deepseek_v3_16b_dist_moe_mxfp8,
    deepseek_v3_16b_minimal_async_ep,
    deepseek_v3_671b,
    deepseek_v3_671b_dist_moe_bf16,
    deepseek_v3_671b_dist_moe_mxfp8,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_minimal_async_ep,
    enable_mlperf_packing,
    fused_mla_query_projection,
)
from torchtitan.trainer import Trainer

from . import model_registry


_DIST_MOE_EXPERT_FQN = "layers.*.moe.routed_experts"
_FUSED_DENSE_SWIGLU_OVERRIDE = "torchtitan.overrides.fused_swiglu.fused_swiglu"


def _graph_dist_moe(
    base_config: Trainer.Config,
) -> GraphTrainer.Config:
    """Convert a complete Standard Trainer DistMoE recipe to GraphTrainer.

    Args:
        base_config: Standard Trainer recipe containing the model policy.

    Returns:
        GraphTrainer recipe with SimpleFSDP and full CUDA graph policy.
    """
    config = to_graph_trainer_config(base_config, model_registry)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=[],
        force_recompute_mm_shapes_by_fqns=[fused_mla_query_projection(config)],
        fsdp_contiguous_module_fqns=[_DIST_MOE_EXPERT_FQN],
        inductor_compilation="none",
        memory_policy="full",
        require_cudagraph=True,
    )
    return config


def _graph_mxfp8_dist_moe(
    base_config: Trainer.Config,
) -> GraphTrainer.Config:
    """Match the Main trainer's non-pipeline MXFP8 execution policies."""
    config = _graph_dist_moe(base_config)

    assert _FUSED_DENSE_SWIGLU_OVERRIDE not in config.override.imports
    config.override.imports.append(_FUSED_DENSE_SWIGLU_OVERRIDE)

    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts

    num_expert_configs = 0
    for _, experts, _, _ in config.traverse(DistMoeRoutedExperts.Config):
        experts.backend = replace(
            experts.backend,
            device_memory_budget_bytes="maximum_useful",
            vmm_host_scratch_imbalance_factor=None,
        )
        num_expert_configs += 1
    assert num_expert_configs > 0
    return config


def graph_trainer_deepseek_v3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_mxfp8() -> GraphTrainer.Config:
    base = deepseek_v3_debugmodel()
    # Quantize dense and moe gemms to mxfp8
    base.model_spec = deepseek_v3_model_registry(
        "debugmodel",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=True,
                fqns=["attention", "shared_experts", "feed_forward"],
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=True,
                pad_multiple=128,
            ),
        ],
    )
    config = to_graph_trainer_config(base, model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_hybridep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def graph_trainer_deepseek_v3_debugmodel_minimal_async_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_debugmodel_minimal_async_ep(),
        model_registry,
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_eager_pp() -> GraphTrainer.Config:
    """Test-only FlexAttention baseline that runs through eager pipeline parallelism."""
    config = graph_trainer_deepseek_v3_debugmodel()
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=["loss"],
        mode=None,
    )
    config.model_spec = replace(config.model_spec, pipelining_fn=pipeline_llm)
    return config


def graph_trainer_deepseek_v3_16b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_16b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the DSV3 16B GraphTrainer config with BF16-compute DistMoE.

    Returns:
        FP32-master DSV3 16B configuration using the standalone BF16 CuTe
        routed experts.
    """
    return _graph_dist_moe(deepseek_v3_16b_dist_moe_bf16())


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the DSV3 16B GraphTrainer config with MXFP8 DistMoE.

    Returns:
        FP32-master DSV3 16B configuration using MXFP8 E4M3 CuTe routed
        experts.
    """
    return _graph_mxfp8_dist_moe(deepseek_v3_16b_dist_moe_mxfp8())


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf() -> (GraphTrainer.Config):
    """Build the optimized 16B MXFP8 recipe with continuous-row packing."""
    config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8()
    enable_mlperf_packing(config)
    config.debug.moe_force_load_balance = True
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_16gpu() -> (
    GraphTrainer.Config
):
    """Build the 16-GPU proxy for the 671B SPMD configuration."""
    config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8()
    assert isinstance(config.loss, ChunkedLossWrapper.Config)
    config.loss = config.loss.loss_fn
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 256 * 4096
    config.activation_checkpoint = None
    config.compile.numerics_changing_optim = True
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"]
    )
    config.dataloader.shuffle = False
    config.dataloader.repeat = True

    parallelism = config.parallelism
    parallelism.data_parallel_replicate_degree = 1
    parallelism.data_parallel_shard_degree = 16
    parallelism.tensor_parallel_degree = 1
    parallelism.context_parallel_degree = 1
    parallelism.pipeline_parallel_degree = 1
    parallelism.expert_parallel_degree = 4
    parallelism.fsdp_reshard_after_forward = "never"

    enable_mlperf_packing(config)
    config.debug.moe_force_load_balance = True
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu() -> (
    GraphTrainer.Config
):
    """Build the DP64/EP64 16B performance proxy with GA16."""
    config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_16gpu()
    config.training.num_tokens_per_train_step = 64 * 16 * 4096
    config.parallelism.data_parallel_shard_degree = 64
    config.parallelism.expert_parallel_degree = 64
    config.compile.inductor_compilation = "regional"
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu_coda() -> (
    GraphTrainer.Config
):
    """Add benchmark-gated CODA fusions to the DP64/EP64 proxy."""
    config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu()
    config.compile.enable_coda = True
    config.compile.coda_patterns = [
        "F_swiglu",
        "B_swiglu_backward_activation",
        "B_parallel_mm_dx_merge",
        "B_mm_dx_residual_add",
        "B_linear_dw_bf16_to_fp32",
    ]
    return config


def graph_trainer_deepseek_v3_16b_minimal_async_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_16b_minimal_async_ep(),
        model_registry,
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_16b_coda() -> GraphTrainer.Config:
    base = deepseek_v3_16b_minimal_async_ep()
    base.hf_assets_path = "./tests/assets/tokenizer"
    base.parallelism = replace(
        base.parallelism,
        data_parallel_shard_degree=2,
        expert_parallel_degree=2,
    )
    base.dataloader.dataset = ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
    base.model_spec = deepseek_v3_model_registry(
        "16B",
        attn_backend="flex_flash",
        moe_comm_backend="minimal_async_ep",
    )
    base.override.imports = [
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
        "torchtitan.overrides.helion_rope.helion_complex_rope",
    ]
    config = to_graph_trainer_config(base, model_registry)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        enable_coda=True,
        inductor_compilation="regional",
        memory_policy="full",
        numerics_changing_optim=True,
    )
    return config


def graph_trainer_deepseek_v3_16b_sdpa() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_16b()
    config.parallelism.context_parallel_load_balancer = "headtail"
    config.model_spec = model_registry("16B", attn_backend="sdpa")
    return config


def graph_trainer_deepseek_v3_671b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_671b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the DSV3 671B GraphTrainer config with BF16-compute DistMoE.

    Returns:
        FP32-master DSV3 671B configuration using the standalone BF16 CuTe
        routed experts.
    """
    return _graph_dist_moe(deepseek_v3_671b_dist_moe_bf16())


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the DSV3 671B GraphTrainer config with MXFP8 DistMoE.

    Returns:
        FP32-master DSV3 671B configuration using MXFP8 E4M3 CuTe routed
        experts.
    """
    return _graph_mxfp8_dist_moe(deepseek_v3_671b_dist_moe_mxfp8())


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf() -> (GraphTrainer.Config):
    """Build the optimized 671B MXFP8 recipe with continuous-row packing."""
    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8()
    enable_mlperf_packing(config)
    config.debug.moe_force_load_balance = True
    return config


def _graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_spmd(
    *,
    local_batch_size: int,
    data_parallel_shard_degree: int,
) -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8()
    assert isinstance(config.loss, ChunkedLossWrapper.Config)
    config.loss = config.loss.loss_fn
    config.training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * 4096
    config.training.num_tokens_per_train_step = 4096 * 4096
    config.activation_checkpoint = None
    config.compile.numerics_changing_optim = True
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"]
    )
    config.dataloader.shuffle = False
    config.dataloader.repeat = True

    parallelism = config.parallelism
    parallelism.data_parallel_replicate_degree = 1
    parallelism.data_parallel_shard_degree = data_parallel_shard_degree
    parallelism.tensor_parallel_degree = 1
    parallelism.context_parallel_degree = 1
    parallelism.pipeline_parallel_degree = 1
    parallelism.expert_parallel_degree = 64

    enable_mlperf_packing(config)
    config.debug.moe_force_load_balance = True
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_64gpu() -> (
    GraphTrainer.Config
):
    """Build the optimized non-pipeline 671B recipe for 64 GPUs."""
    return _graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_spmd(
        local_batch_size=1,
        data_parallel_shard_degree=64,
    )


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_256gpu() -> (
    GraphTrainer.Config
):
    """Build the DP256/EP64 671B recipe with eFSDP4 and LBS2/GA8."""
    return _graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_spmd(
        local_batch_size=2,
        data_parallel_shard_degree=256,
    )
