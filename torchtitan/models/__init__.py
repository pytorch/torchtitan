# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect

from torchtitan.config import Configurable

_supported_models = frozenset(
    [
        "deepseek_v3",
        "deepseek_v4",
        "flux",
        "gpt_oss",
        "kimi_k2_7",
        "kimi_k3",
        "llama3",
        "muse_glimmer",
        "qwen3",
        "qwen3_5",
        "qwen3_6",
        "qwen3_8",
    ]
)


def build_model_config(
    model_name: str, model_flavor: str, *, enable_sp: bool
) -> Configurable.Config:
    """Build a named model config with its construction-time SP setting."""
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_registry = model_module.model_registry
    registry_kwargs = {}
    if "enable_sp" in inspect.signature(model_registry).parameters:
        registry_kwargs["enable_sp"] = enable_sp
    return model_registry(model_flavor, **registry_kwargs)
