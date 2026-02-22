# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConvertersContainer


def build_parallel_dims(trainer_config, world_size):
    parallelism_config = trainer_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )
    return parallel_dims


def test_build_model_converters_empty_list():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    parallel_dims = build_parallel_dims(config, 1)

    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    model_converters = config.model_converters.build(
        parallel_dims=parallel_dims,
        model_compile_enabled=model_compile_enabled,
    )
    assert isinstance(model_converters, ModelConvertersContainer)
    assert model_converters.converters == []


def test_build_model_converters_float8_converter():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    # Set converter config directly (not via CLI)
    config.model_converters = ModelConvertersContainer.Config(
        converters=[Float8LinearConverter.Config(emulate=True)],
    )
    parallel_dims = build_parallel_dims(config, 1)

    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    model_converters = config.model_converters.build(
        parallel_dims=parallel_dims,
        model_compile_enabled=model_compile_enabled,
    )
    assert isinstance(model_converters, ModelConvertersContainer)
    assert len(model_converters.converters) == 1
    assert isinstance(model_converters.converters[0], Float8LinearConverter)
