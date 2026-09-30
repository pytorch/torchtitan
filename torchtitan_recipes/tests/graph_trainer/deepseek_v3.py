# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 GraphTrainer configurations used by tests."""

import os
from dataclasses import replace

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.dist_moe import DistMoeRoutedExperts
from torchtitan.config.transform import (
    apply_transforms,
    MXFP8GroupedLinearConverter,
    TokenDispatcherTransform,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)

from torchtitan.experiments.graph_trainer.deepseek_v3 import build_model_config
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)

from torchtitan_recipes.models.deepseek_v3 import (
    configure_deepseek_v3_round_robin_routing,
    deepseek_v3_671b_dist_moe_bf16,
    deepseek_v3_671b_dist_moe_mxfp8,
)
from torchtitan_recipes.tests.models.deepseek_v3 import (
    configure_deepseek_v3_16b_dist_moe_local_4gpu,
    deepseek_v3_16b_dist_moe_bf16,
    deepseek_v3_16b_dist_moe_mxfp8,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_dist_moe_bf16,
    deepseek_v3_debugmodel_dist_moe_mxfp8,
    deepseek_v3_mxfp8_linear_converter_config,
)


def _dist_moe_graph_config(base) -> GraphTrainer.Config:
    """Convert one eager Dist-MoE recipe to the GraphTrainer model wrapper."""
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_debugmodel(),
        GraphTrainerDeepSeekV3Model.Config,
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_debugmodel_mxfp8() -> GraphTrainer.Config:
    base = deepseek_v3_debugmodel()
    # Quantize dense and moe gemms to mxfp8
    base.model = build_deepseek_v3_model_config(
        "debugmodel",
        seq_len=base.training.max_context_length,
        converters=[
            deepseek_v3_mxfp8_linear_converter_config(
                model_compile_enabled=True,
            ),
            MXFP8GroupedLinearConverter.Config(
                model_compile_enabled=True,
                pad_multiple=128,
            ),
        ],
    )
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the GraphTrainer debug recipe with BF16 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_debugmodel_dist_moe_bf16(seq_len=2048))


def graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the GraphTrainer debug recipe with MXFP8 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_debugmodel_dist_moe_mxfp8(seq_len=2048))


def graph_trainer_deepseek_v3_debugmodel_hybridep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_debugmodel(), GraphTrainerDeepSeekV3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    config.parallelism.expert_parallel_degree = 2
    config.model = build_model_config(
        "debugmodel",
        seq_len=config.training.max_context_length,
    )
    return apply_transforms(
        config,
        [
            TokenDispatcherTransform(
                dispatcher=HybridEPTokenDispatcher,
                kwargs={"non_blocking_capacity_factor": 1.0},
            )
        ],
    )


def _graph_trainer_deepseek_v3_16b_dist_moe(
    *,
    mxfp8: bool,
) -> GraphTrainer.Config:
    """Build the matched four-GPU DistMoE performance configuration."""
    base = (
        deepseek_v3_16b_dist_moe_mxfp8(
            seq_len=4096,
            device_scratch_capacity_factor=1.0,
        )
        if mxfp8
        else deepseek_v3_16b_dist_moe_bf16(
            seq_len=4096,
            device_scratch_capacity_factor=1.0,
        )
    )
    base = configure_deepseek_v3_16b_dist_moe_local_4gpu(base)
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.loss = config.loss.loss_fn
    config.compile = GraphTrainerCompileConfig(
        fsdp_param_unshard_mode="extracted_in_schedule_stage",
        fsdp_gradient_sync_mode="deferred_as_schedule_stage",
        gradient_accum_in_wgrad_fusion="enabled",
        memory_policy="none",
        inductor_compilation="regional",
        numerics_changing_optim=True,
        enable_fsdp_ag_rs_overlap=True,
        enable_fsdp_dense_region_overlap=False,
    )
    if os.environ.get("CODA_BOUNDARY_MODE", "schedule") == "edge":
        config.compile.fsdp_param_unshard_mode = "only_in_first_microbatch"
        config.compile.fsdp_gradient_sync_mode = "only_in_last_microbatch"
    config.training.disable_cuda_graphs = False
    fused_swiglu = "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu_feed_forward"
    if fused_swiglu not in config.override.imports:
        config.override.imports.append(fused_swiglu)
    fused_mla = "torchtitan_recipes.overrides.fused_mla.fused_mla"
    if fused_mla not in config.override.imports:
        config.override.imports.append(fused_mla)
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_bf16_local_4gpu() -> (GraphTrainer.Config):
    return _graph_trainer_deepseek_v3_16b_dist_moe(mxfp8=False)


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_local_4gpu() -> (GraphTrainer.Config):
    return _graph_trainer_deepseek_v3_16b_dist_moe(mxfp8=True)


def graph_trainer_deepseek_v3_671b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 671B recipe with BF16 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_bf16(seq_len=4096))


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 671B recipe with MXFP8 Dist-MoE experts."""
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

    config.loss = config.loss.loss_fn
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 4096 * 4096
    config.training.steps = 60
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
    configure_deepseek_v3_round_robin_routing(config)
    config.metrics.log_freq = 10
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
