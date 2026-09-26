# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect

from torchtitan.config import Configurable


def build_model_config_for_conversion(
    model_name: str, model_flavor: str
) -> Configurable.Config:
    """Build the unparallelized model config used for checkpoint conversion."""
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_registry = model_module.model_registry
    registry_kwargs = {}
    if "enable_sp" in inspect.signature(model_registry).parameters:
        # Conversion only needs parameter shapes and FQNs, which are identical
        # for the SP and non-SP shared-expert implementations.
        registry_kwargs["enable_sp"] = False
    return model_registry(model_flavor, **registry_kwargs)
