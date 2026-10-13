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
        flavor, converters=[_ReplaceRootConverter.Config()]
    , parallelism=ParallelismConfig())

    assert _ReplaceRootConverter.returned
    assert config is _ReplaceRootConverter.returned[-1]
