# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.float8 import Float8Converter
from torchtitan.config_manager import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    build_model_converters,
    ModelConvertersContainer,
)


def build_parallel_dims(job_config, world_size):
    parallelism_config = job_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not parallelism_config.disable_loss_parallel,
    )
    return parallel_dims


def test_build_model_converters_empty_list():
    config_manager = ConfigManager()
    config = config_manager.parse_args([])
    parallel_dims = build_parallel_dims(config, 1)

    model_converters = build_model_converters(config, parallel_dims)
    assert isinstance(model_converters, ModelConvertersContainer)
    assert model_converters.converters == []


def test_build_model_converters_float8_converter():
    config_manager = ConfigManager()
    config = config_manager.parse_args(["--model.converters", "float8"])
    parallel_dims = build_parallel_dims(config, 1)

    model_converters = build_model_converters(config, parallel_dims)
    assert isinstance(model_converters, ModelConvertersContainer)
    assert len(model_converters.converters) == 1
    assert isinstance(model_converters.converters[0], Float8Converter)
