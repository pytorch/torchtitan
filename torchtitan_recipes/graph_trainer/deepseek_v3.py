# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified DeepSeek V3 GraphTrainer recipes."""

from dataclasses import replace

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.dist_moe import DistMoeRoutedExperts
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.observability.profiler import Profiler

from torchtitan_recipes.models.deepseek_v3 import (
    configure_deepseek_v3_round_robin_routing,
    deepseek_v3_16b,
    deepseek_v3_16b_dist_moe_bf16,
    deepseek_v3_16b_dist_moe_mxfp8,
    deepseek_v3_671b_dist_moe_bf16,
    deepseek_v3_671b_dist_moe_mxfp8,
)


def _dist_moe_graph_config(base) -> GraphTrainer.Config:
    """Convert one eager DistMoE recipe to the GraphTrainer model wrapper."""
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_16b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_16b(seq_len=4096), GraphTrainerDeepSeekV3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 16B recipe with BF16 DistMoE experts."""
    return _dist_moe_graph_config(deepseek_v3_16b_dist_moe_bf16(seq_len=4096))


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 16B recipe with MXFP8 DistMoE experts."""
    return _dist_moe_graph_config(deepseek_v3_16b_dist_moe_mxfp8(seq_len=4096))


def graph_trainer_deepseek_v3_671b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 671B recipe with BF16 DistMoE experts."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_bf16(seq_len=4096))


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 671B recipe with MXFP8 DistMoE experts."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_mxfp8(seq_len=4096))


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu() -> (
    GraphTrainer.Config
):
    """Build Chien-Chin's 256-GPU ladder1 R4 configuration."""
    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8()
    if not isinstance(config.dataloader, GrainDataLoader.Config):
        raise TypeError("the Chien-Chin recipe requires GrainDataLoader")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"],
        mask_document_boundaries=False,
    )
    config.dataloader.shuffle = False
    config.dataloader.repeat = True
    config.dataloader.max_num_documents = 1
    for _, inner_attention, _, _ in config.model.traverse(VarlenInnerAttention.Config):
        inner_attention.fixed_length_rows = True
    for _, experts, _, _ in config.model.traverse(DistMoeRoutedExperts.Config):
        experts.backend = replace(
            experts.backend,
            device_scratch_capacity_factor=1.0,
        )
    configure_deepseek_v3_round_robin_routing(config)

    if not isinstance(config.loss, ChunkedLossWrapper.Config):
        raise TypeError("the Chien-Chin recipe requires ChunkedLossWrapper")
    config.loss = config.loss.loss_fn
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 4096 * 4096
    config.training.steps = 40
    config.training.disable_cuda_graphs = False
    config.activation_checkpoint = None
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 256
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_symm_mem_scope = "dense"
    config.metrics.log_freq = 1
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config.compile = GraphTrainerCompileConfig(
        fsdp_param_unshard_mode="only_in_first_microbatch",
        fsdp_gradient_sync_mode="only_in_last_microbatch",
        gradient_accum_in_wgrad_fusion="enabled",
        memory_policy="none",
        inductor_compilation="regional",
        numerics_changing_optim=False,
        enable_fsdp_ag_rs_overlap=True,
        enable_fsdp_dense_region_overlap=False,
    )
    for override in (
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu_feed_forward",
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
    ):
        if override not in config.override.imports:
            config.override.imports.append(override)
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile() -> (
    GraphTrainer.Config
):
    """Profile step 41 of the Chien-Chin 256-GPU R4 configuration."""
    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu()
    config.training.steps = 41
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=41,
        profiler_warmup=0,
        profiler_active=1,
        profiler_repeat=1,
        enable_memory_snapshot=True,
        memory_snapshot_freq=41,
    )
    return config
