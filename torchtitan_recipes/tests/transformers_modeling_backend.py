# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the Transformers modeling backend integration tests."""

from torchtitan.experiments.transformers_modeling_backend.config_registry import (
    transformers_modeling_backend_debugmodel,
    transformers_modeling_backend_debugmodel_moe,
    transformers_modeling_backend_sft_debugmodel,
    TransformersBackendConfig,
)


def transformers_backend_moe_fsdp_tp_ep_cp() -> TransformersBackendConfig:
    config = transformers_modeling_backend_debugmodel_moe()
    config.parallelism.data_parallel_shard_degree = -1
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = "ptrr"
    config.training.disable_cuda_graphs = True
    config.training.steps = 2
    return config


def transformers_backend_dense_fsdp_tp_pp() -> TransformersBackendConfig:
    config = transformers_modeling_backend_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    config.training.steps = 2
    return config


def transformers_backend_dense_cp_pp() -> TransformersBackendConfig:
    config = transformers_modeling_backend_debugmodel()
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.context_parallel_load_balancer = "ptrr"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    config.training.steps = 2
    return config


def transformers_backend_sft() -> TransformersBackendConfig:
    config = transformers_modeling_backend_sft_debugmodel()
    config.training.steps = 2
    return config
