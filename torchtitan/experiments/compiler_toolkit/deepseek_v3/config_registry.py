# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.compiler_toolkit.configs import (
    CompilerToolkitCompileConfig,
    to_compiler_toolkit_config,
)
from torchtitan.experiments.compiler_toolkit.trainer import CompilerToolkitTrainer
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_flex_attn,
)

from . import model_registry


def compiler_toolkit_deepseek_v3_debugmodel() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = CompilerToolkitCompileConfig()
    return config


def compiler_toolkit_deepseek_v3_debugmodel_flex_attn() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(
        deepseek_v3_debugmodel_flex_attn(), model_registry
    )
    config.compile = CompilerToolkitCompileConfig()
    return config


def compiler_toolkit_deepseek_v3_16b() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(deepseek_v3_16b(), model_registry)
    config.compile = CompilerToolkitCompileConfig()
    return config


def compiler_toolkit_deepseek_v3_671b() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(deepseek_v3_671b(), model_registry)
    config.compile = CompilerToolkitCompileConfig()
    return config
