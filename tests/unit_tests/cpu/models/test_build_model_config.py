# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts shared by every model's ``build_model_config``."""

import copy
import importlib
from dataclasses import dataclass

import pytest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.models.common.attention import InnerAttention, VarlenInnerAttention
from torchtitan.models.common.decoder import Decoder

_MODELS = [
    "llama3",
    "qwen3",
    "qwen3_5",
    "qwen3_6",
    "qwen3_8",
    "deepseek_v3",
    "deepseek_v4",
    "gpt_oss",
    "kimi_k2_7",
    "kimi_k3",
    "muse_glimmer",
    "flux",
]


@pytest.mark.parametrize(
    "model", [name for name in _MODELS if name not in ("deepseek_v4", "kimi_k2_7")]
)
def test_model_defaults_to_varlen(model):
    module = importlib.import_module(f"torchtitan.models.{model}")
    config = module.build_model_config(
        next(iter(module.MODEL_FLAVORS)), parallelism=ParallelismConfig()
    )
    layers = (
        [layer.attention for layer in config.layers if layer.attention is not None]
        if isinstance(config, Decoder.Config)
        else [config]
    )
    attention_configs = [
        attention
        for layer in layers
        for _, attention, _, _ in layer.traverse(InnerAttention.Config)
    ]
    assert attention_configs
    assert all(
        isinstance(attention, VarlenInnerAttention.Config)
        for attention in attention_configs
    )


class _ReplaceRootConverter(ModelConfigConverter):
    """Return a new root config and remember it."""

    returned: list = []

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config):
        replacement = copy.copy(model_config)
        _ReplaceRootConverter.returned.append(replacement)
        return replacement


@pytest.mark.parametrize("model", _MODELS)
def test_build_model_config_returns_the_converted_root(model):
    """A converter may replace the root config; the builder must return it."""
    module = importlib.import_module(f"torchtitan.models.{model}")
    flavor = next(iter(module.MODEL_FLAVORS))
    _ReplaceRootConverter.returned.clear()

    config = module.build_model_config(
        flavor,
        parallelism=ParallelismConfig(),
        converters=[_ReplaceRootConverter.Config()],
    )

    assert _ReplaceRootConverter.returned
    assert config is _ReplaceRootConverter.returned[-1]
