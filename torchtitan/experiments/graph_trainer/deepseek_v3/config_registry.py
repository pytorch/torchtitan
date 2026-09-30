# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config.transform import MXFP8GroupedLinearConverter
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_16b_nvfp4,
    deepseek_v3_671b,
    deepseek_v3_671b_nvfp4_mixed,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_nvfp4,
    deepseek_v3_mxfp8_linear_converter_config,
)
from torchtitan.trainer import Trainer

from . import model_registry
from .model import GraphTrainerDeepSeekV3Model


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


def _to_nvfp4_graph_trainer_config(base: Trainer.Config) -> GraphTrainer.Config:
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig(
        inductor_compilation="full",
        numerics_changing_optim=False,
        memory_policy="full",
        disable_passes=["cuda_graph_pass"],
    )
    config.training.disable_cuda_graphs = True
    return config


def graph_trainer_deepseek_v3_debugmodel_nvfp4(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> GraphTrainer.Config:
    return _to_nvfp4_graph_trainer_config(deepseek_v3_debugmodel_nvfp4(seq_len=seq_len))


def graph_trainer_deepseek_v3_16b_nvfp4(
    bf16_tail_fraction: float = 0.0,
    *,
    seq_len: int | None = 4096,
) -> GraphTrainer.Config:
    """Cluster validation candidate; only the debugmodel has a numerical gate."""
    return _to_nvfp4_graph_trainer_config(
        deepseek_v3_16b_nvfp4(bf16_tail_fraction, seq_len=seq_len)
    )


def graph_trainer_deepseek_v3_671b_nvfp4_mixed(
    seq_len: int | None = 4096,
) -> GraphTrainer.Config:
    """Cluster validation candidate; only the debugmodel has a numerical gate."""
    return _to_nvfp4_graph_trainer_config(deepseek_v3_671b_nvfp4_mixed(seq_len=seq_len))


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


def graph_trainer_deepseek_v3_16b_sdpa() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_16b()
    config.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
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
