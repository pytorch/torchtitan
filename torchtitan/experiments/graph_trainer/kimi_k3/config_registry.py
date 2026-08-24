# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import torch._inductor.config as inductor_config

from torchtitan.components.data import GrainDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

from . import model_registry
from .data import KimiK3TextProcessor


def graph_trainer_kimi_k3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(kimi_k3_debugmodel(), model_registry)
    config.model_spec = model_registry("debugmodel")
    config.tokenizer = HuggingFaceTokenizer.Config()
    config.dataloader = GrainDataLoader.Config(
        dataset=replace(
            DATASETS["c4_test"],
            processor=KimiK3TextProcessor.Config(),
        ),
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_kimi_k3_16b() -> GraphTrainer.Config:
    config = graph_trainer_kimi_k3_debugmodel()
    config.model_spec = model_registry("16B")
    return config


def graph_trainer_kimi_k3_16b_text() -> GraphTrainer.Config:
    """Alias for the language-only 16B GraphTrainer recipe."""
    config = graph_trainer_kimi_k3_debugmodel()
    config.model_spec = model_registry("16B-text")
    return config


def graph_trainer_kimi_k3_15b_compute_bound() -> GraphTrainer.Config:
    """Build the largest Kimi K3 batch-16 recipe that fits on two GB200 GPUs."""
    inductor_config.shape_padding = True
    inductor_config.force_shape_pad = True

    config = graph_trainer_kimi_k3_debugmodel()
    config.model_spec = model_registry("15B-compute-bound")
    config.parallelism = replace(
        config.parallelism,
        data_parallel_shard_degree=2,
    )
    config.training = replace(
        config.training,
        num_tokens_per_microbatch_per_dp_rank=65536,
        max_context_length=4096,
        disable_cuda_graphs=False,
    )
    config.debug.moe_force_load_balance = True
    config.comm.init_timeout_seconds = 1800
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=[],
        inductor_compilation="full",
        memory_policy="full",
        require_cudagraph=True,
    )
    return config
