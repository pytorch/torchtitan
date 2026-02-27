# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.experiments.fullmodel.configs import (
    FullmodelCompileConfig,
    to_fullmodel_config,
)
from torchtitan.experiments.fullmodel.trainer import FullmodelTrainer
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_flex_attn,
)

from . import model_registry


def fullmodel_deepseek_v3_debugmodel() -> FullmodelTrainer.Config:
    config = to_fullmodel_config(deepseek_v3_debugmodel(), model_registry)
    config.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    config.compile = FullmodelCompileConfig(enable=True)
    return config


def fullmodel_deepseek_v3_debugmodel_flex_attn() -> (FullmodelTrainer.Config):
    config = to_fullmodel_config(deepseek_v3_debugmodel_flex_attn(), model_registry)
    config.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    config.compile = FullmodelCompileConfig(enable=True)
    return config


def fullmodel_deepseek_v3_16b() -> FullmodelTrainer.Config:
    config = to_fullmodel_config(deepseek_v3_16b(), model_registry)
    config.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    config.compile = FullmodelCompileConfig(enable=True)
    return config


def fullmodel_deepseek_v3_671b() -> FullmodelTrainer.Config:
    config = to_fullmodel_config(deepseek_v3_671b(), model_registry)
    config.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    config.compile = FullmodelCompileConfig(enable=True)
    return config
