# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.fullmodel.configs import (
    FullmodelCompileConfig,
    to_fullmodel_config,
)
from torchtitan.experiments.fullmodel.trainer import FullmodelTrainer
from torchtitan.models.llama3.config_registry import (
    llama3_405b,
    llama3_70b,
    llama3_8b,
    llama3_debugmodel,
    llama3_debugmodel_flex_attn,
)

from . import model_registry


def fullmodel_llama3_debugmodel() -> FullmodelTrainer.Config:
    config = to_fullmodel_config(llama3_debugmodel(), model_registry)
    config.compile = FullmodelCompileConfig(enable=True)
    return config


def fullmodel_llama3_debugmodel_flex_attn() -> (FullmodelTrainer.Config):
    config = to_fullmodel_config(llama3_debugmodel_flex_attn(), model_registry)
    config.compile = FullmodelCompileConfig(enable=True)
    return config


def fullmodel_llama3_8b() -> FullmodelTrainer.Config:
    config = to_fullmodel_config(llama3_8b(), model_registry)
    config.compile = FullmodelCompileConfig(enable=True)
    return config


def fullmodel_llama3_70b() -> FullmodelTrainer.Config:
    config = to_fullmodel_config(llama3_70b(), model_registry)
    config.compile = FullmodelCompileConfig(enable=True)
    return config


def fullmodel_llama3_405b() -> FullmodelTrainer.Config:
    config = to_fullmodel_config(llama3_405b(), model_registry)
    config.compile = FullmodelCompileConfig(enable=True)
    return config
