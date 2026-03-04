# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.experiments.simple_fsdp.configs import (
    SimpleFSDPCompileConfig,
    SimpleFSDPConfig,
    to_simple_fsdp_config,
)
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_flex_attn,
)

from . import model_registry


def simple_fsdp_deepseek_v3_debugmodel() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(deepseek_v3_debugmodel(), model_registry)
    config.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    config.compile = SimpleFSDPCompileConfig(
        enable=True,
        graph_passes="auto_bucketing",
    )
    return config


def simple_fsdp_deepseek_v3_debugmodel_flex_attn() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(deepseek_v3_debugmodel_flex_attn(), model_registry)
    config.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    config.compile = SimpleFSDPCompileConfig(
        enable=True,
        graph_passes="auto_bucketing",
    )
    return config


def simple_fsdp_deepseek_v3_16b() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(deepseek_v3_16b(), model_registry)
    config.compile = SimpleFSDPCompileConfig(enable=True)
    return config


def simple_fsdp_deepseek_v3_671b() -> SimpleFSDPConfig:
    config = to_simple_fsdp_config(deepseek_v3_671b(), model_registry)
    config.compile = SimpleFSDPCompileConfig(enable=True)
    return config
