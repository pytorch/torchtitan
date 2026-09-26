# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config.transform import MXFP8GroupedLinearConverter

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
    deepseek_v3_671b,
    deepseek_v3_671b_dist_moe_bf16,
    deepseek_v3_671b_dist_moe_mxfp8,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_dist_moe_bf16,
    deepseek_v3_debugmodel_dist_moe_mxfp8,
    deepseek_v3_mxfp8_linear_converter_config,
)

from . import model_registry
from .model import GraphTrainerDeepSeekV3Model


def _dist_moe_graph_config(base) -> GraphTrainer.Config:
    """Convert one eager Dist-MoE recipe to the GraphTrainer model wrapper."""
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_debugmodel(), GraphTrainerDeepSeekV3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_debugmodel_mxfp8() -> GraphTrainer.Config:
    base = deepseek_v3_debugmodel()
    # Quantize dense and moe gemms to mxfp8
    base.model = deepseek_v3_model_registry(
        "debugmodel",
        enable_sp=True,
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
    config.model = model_registry(
        "debugmodel",
        enable_sp=True,
        seq_len=config.training.max_context_length,
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def graph_trainer_deepseek_v3_16b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_16b(seq_len=4096), GraphTrainerDeepSeekV3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
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
    config.model = model_registry(
        "16B",
        enable_sp=True,
        seq_len=config.training.max_context_length,
        attn_backend="sdpa",
    )
    return config


def graph_trainer_deepseek_v3_671b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_671b(seq_len=4096), GraphTrainerDeepSeekV3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 671B recipe with BF16 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_bf16(seq_len=4096))


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the GraphTrainer DSV3 671B recipe with MXFP8 Dist-MoE experts."""
    return _dist_moe_graph_config(deepseek_v3_671b_dist_moe_mxfp8(seq_len=4096))
