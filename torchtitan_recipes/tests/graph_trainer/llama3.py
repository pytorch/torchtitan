# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama 3 GraphTrainer configurations used by tests."""

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3 import (
    build_model_config as build_graph_llama3_model_config,
)

from torchtitan.experiments.graph_trainer.llama3.model import GraphTrainerLlama3Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import build_model_config as build_llama3_model_config
from torchtitan.observability.sdc_replayer import SDCReplayer

from torchtitan_recipes.models.llama3 import llama3_mxfp8_linear_converter_config
from torchtitan_recipes.tests.models.llama3 import (
    llama3_debugmodel,
    llama3_debugmodel_varlen_attn,
)


def graph_trainer_llama3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        llama3_debugmodel(), GraphTrainerLlama3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_debugmodel_sdc_replay() -> GraphTrainer.Config:
    config = graph_trainer_llama3_debugmodel()
    config.debug.deterministic = True
    config.debug.seed = 42
    config.training.disable_cuda_graphs = True
    config.training.steps = 2
    config.sdc_replayer = SDCReplayer.Config()
    return config


def graph_trainer_llama3_debugmodel_mxfp8() -> GraphTrainer.Config:
    base = llama3_debugmodel()
    base.model = build_llama3_model_config(
        "debugmodel_mxfp8",
        seq_len=base.training.max_context_length,
        converters=[
            llama3_mxfp8_linear_converter_config(),
        ],
        parallelism=base.parallelism,
    )
    config = to_graph_trainer_config(base, GraphTrainerLlama3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_debugmodel_varlen() -> GraphTrainer.Config:
    """Debug model using Varlen attention for graph machinery tests."""
    base = llama3_debugmodel_varlen_attn()
    base.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
    base.model = build_graph_llama3_model_config(
        "debugmodel",
        seq_len=base.training.max_context_length,
        attn_backend="varlen",
        parallelism=base.parallelism,
    )
    config = to_graph_trainer_config(base, GraphTrainerLlama3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_debugmodel_varlen_cross_entropy_loss() -> GraphTrainer.Config:
    """Varlen debug model with standard cross-entropy loss."""
    config = graph_trainer_llama3_debugmodel_varlen()
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model),
    )
    return config
