# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.quantization.mx import MXFP8Converter
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.llama3.config_registry import (
    llama3_405b,
    llama3_70b,
    llama3_8b,
    llama3_debugmodel,
    llama3_debugmodel_flex_attn,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer

from . import model_registry


def graph_trainer_llama3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(llama3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_llama3_debugmodel_flex_attn() -> (GraphTrainer.Config):
    config = to_graph_trainer_config(llama3_debugmodel_flex_attn(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_llama3_debugmodel_mxfp8() -> GraphTrainer.Config:
    """Llama3 debugmodel with MXFP8 quantization. Requires SM100+ (B200/B100)
    and torchao."""
    config = to_graph_trainer_config(llama3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            MXFP8Converter.Config(fqns=["layers"]),
        ],
    )
    return config


def graph_trainer_llama3_8b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(llama3_8b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_llama3_8b_mxfp8() -> GraphTrainer.Config:
    """Llama3 8B with MXFP8 quantization. Requires SM100+ (B200/B100)
    and torchao."""
    config = to_graph_trainer_config(llama3_8b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            MXFP8Converter.Config(fqns=["layers"]),
        ],
    )
    return config


def graph_trainer_llama3_70b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(llama3_70b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_llama3_405b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(llama3_405b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config
