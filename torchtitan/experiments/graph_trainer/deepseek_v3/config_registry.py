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
    deepseek_v3_16b_minimal_async_ep,
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_minimal_async_ep,
)

from . import model_registry


def graph_trainer_deepseek_v3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_mxfp8() -> GraphTrainer.Config:
    base = deepseek_v3_debugmodel()
    # Quantize the MoE expert grouped GEMMs to MXFP8 on the base model config
    # before wrapping in the graph_trainer config. to_graph_trainer_config copies
    # the (now MXFP8-swapped) model fields into the graph model class and swaps in
    # graph_trainer's parallelize_fn. graph_trainer always compiles the model, so
    # the converter's compile requirement is satisfied. pad_multiple=128 is
    # required by the CuTeDSL quantization kernel on sm_100 (e.g. B200); the
    # default of 32 only suffices on older architectures.
    base.model_spec = deepseek_v3_model_registry(
        "debugmodel",
        converters=[
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


def graph_trainer_deepseek_v3_16b_minimal_async_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_16b_minimal_async_ep(),
        model_registry,
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_16b_sdpa() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_16b()
    config.model_spec = model_registry("16B", attn_backend="sdpa")
    return config


def graph_trainer_deepseek_v3_671b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_671b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config
