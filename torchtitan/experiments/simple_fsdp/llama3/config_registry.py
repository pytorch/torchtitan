# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.simple_fsdp.configs import (
    SimpleFSDPCompileConfig,
    SimpleFSDPConfig,
    to_simple_fsdp_config,
)
from torchtitan.models.llama3.config_registry import (
    llama3_405b,
    llama3_70b,
    llama3_8b,
    llama3_debugmodel,
    llama3_debugmodel_flex_attn,
)

from . import model_registry


def simple_fsdp_llama3_debugmodel() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(llama3_debugmodel(), model_registry)
    config.compile = SimpleFSDPCompileConfig(enable=True)
    return config


def simple_fsdp_llama3_debugmodel_flex_attn() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(llama3_debugmodel_flex_attn(), model_registry)
    config.compile = SimpleFSDPCompileConfig(enable=True)
    return config


def simple_fsdp_llama3_8b() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(llama3_8b(), model_registry)
    config.compile = SimpleFSDPCompileConfig(enable=True)
    return config


def simple_fsdp_llama3_70b() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(llama3_70b(), model_registry)
    config.compile = SimpleFSDPCompileConfig(enable=True)
    return config


def simple_fsdp_llama3_405b() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(llama3_405b(), model_registry)
    config.compile = SimpleFSDPCompileConfig(enable=True)
    return config
