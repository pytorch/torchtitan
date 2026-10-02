# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Llama 3 GraphTrainer recipes."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3.model import GraphTrainerLlama3Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS

from torchtitan_recipes.models.llama3 import llama3_8b


def graph_trainer_llama3_8b(*, seq_len: int = 8192) -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        llama3_8b(seq_len=seq_len), GraphTrainerLlama3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_llama3_8b_c4_test() -> GraphTrainer.Config:
    config = graph_trainer_llama3_8b()
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    return config


def graph_trainer_llama3_8b_autoresearch() -> GraphTrainer.Config:
    config = graph_trainer_llama3_8b_c4_test()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.steps = 20
    config.metrics.enable_tensorboard = False
    config.profiler.enable_profiling = False
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    return config
