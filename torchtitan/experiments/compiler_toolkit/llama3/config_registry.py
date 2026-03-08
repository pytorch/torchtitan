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
from torchtitan.models.llama3.config_registry import (
    llama3_405b,
    llama3_70b,
    llama3_8b,
    llama3_debugmodel,
    llama3_debugmodel_flex_attn,
)

from . import model_registry


def compiler_toolkit_llama3_debugmodel() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(llama3_debugmodel(), model_registry)
    config.compile = CompilerToolkitCompileConfig()
    return config


def compiler_toolkit_llama3_debugmodel_flex_attn() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(llama3_debugmodel_flex_attn(), model_registry)
    config.compile = CompilerToolkitCompileConfig()
    return config


def compiler_toolkit_llama3_8b() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(llama3_8b(), model_registry)
    config.compile = CompilerToolkitCompileConfig()
    return config


def compiler_toolkit_llama3_70b() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(llama3_70b(), model_registry)
    config.compile = CompilerToolkitCompileConfig()
    return config


def compiler_toolkit_llama3_405b() -> CompilerToolkitTrainer.Config:
    config = to_compiler_toolkit_config(llama3_405b(), model_registry)
    config.compile = CompilerToolkitCompileConfig()
    return config
