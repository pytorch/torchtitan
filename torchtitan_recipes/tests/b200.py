# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _use_spmd_types


def kimi_k3_debugmodel_mm_fsdp2() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    return config


def llama3_debugmodel_mxfp8_fsdp2() -> Trainer.Config:
    from torchtitan.models.llama3.config_registry import llama3_debugmodel_mxfp8

    config = llama3_debugmodel_mxfp8()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def graph_trainer_llama3_debugmodel_mxfp8_fsdp2_pp2() -> Trainer.Config:
    from torchtitan.experiments.graph_trainer.llama3.config_registry import (
        graph_trainer_llama3_debugmodel_mxfp8,
    )

    config = graph_trainer_llama3_debugmodel_mxfp8()
    config.compile.disable_passes = ["cudagraph_pass"]
    config.compile.enable_fsdp_dense_region_overlap = True
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 4
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
