# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.quantization import MXFP8GroupedExpertsConverter
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
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
    deepseek_v3_debugmodel_dist_moe_bf16,
    deepseek_v3_debugmodel_dist_moe_mxfp8,
    deepseek_v3_debugmodel_minimal_async_ep,
    deepseek_v3_mxfp8_linear_converter_config,
)

from . import model_registry


def _dist_moe_graph_config(base) -> GraphTrainer.Config:
    """Convert one eager Dist-MoE recipe to the GraphTrainer model wrapper."""
    config = to_graph_trainer_config(base, model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
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
        seq_len=base.training.max_context_length,
        converters=[
            deepseek_v3_mxfp8_linear_converter_config(
                model_compile_enabled=True,
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


def graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the GraphTrainer debug recipe with BF16 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_debugmodel_dist_moe_bf16(seq_len=2048))


def graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the GraphTrainer debug recipe with MXFP8 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_debugmodel_dist_moe_mxfp8(seq_len=2048))


def graph_trainer_deepseek_v3_debugmodel_hybridep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    config.model_spec = model_registry(
        "debugmodel",
        seq_len=config.training.max_context_length,
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
    config = to_graph_trainer_config(deepseek_v3_16b(seq_len=4096), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_16b_minimal_async_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_16b_minimal_async_ep(seq_len=4096),
        model_registry,
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 16B recipe with BF16 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_16b_dist_moe_bf16(seq_len=4096))


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 16B recipe with MXFP8 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_16b_dist_moe_mxfp8(seq_len=4096))


def graph_trainer_deepseek_v3_16b_sdpa() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_16b()
    config.parallelism.context_parallel_load_balancer = "headtail"
    config.model_spec = model_registry(
        "16B",
        seq_len=config.training.max_context_length,
        attn_backend="sdpa",
    )
    return config


def graph_trainer_deepseek_v3_671b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_671b(seq_len=4096), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 671B recipe with BF16 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_bf16(seq_len=4096))


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 671B recipe with MXFP8 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_mxfp8(seq_len=4096))
