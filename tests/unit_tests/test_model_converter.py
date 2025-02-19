# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config_manager import JobConfig
from torchtitan.float8 import Float8Converter
from torchtitan.model_converter import build_model_converters, ModelConvertersContainer
from torchtitan.parallelisms import ParallelDims


def build_parallel_dims(job_config, world_size):
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    return parallel_dims


def test_build_model_converters_empty_list():
    config = JobConfig()
    config.parse_args([])
    parallel_dims = build_parallel_dims(config, 1)

    model_converters = build_model_converters(config, parallel_dims)
    assert isinstance(model_converters, ModelConvertersContainer)
    assert model_converters.converters == []


def test_build_model_converters_float8_converter():
    config = JobConfig()
    config.parse_args(["--model.converters", "float8"])
    parallel_dims = build_parallel_dims(config, 1)

    model_converters = build_model_converters(config, parallel_dims)
    assert isinstance(model_converters, ModelConvertersContainer)
    assert len(model_converters.converters) == 1
    assert isinstance(model_converters.converters[0], Float8Converter)
