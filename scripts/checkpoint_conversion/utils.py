# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

from torchtitan.config import Configurable


def build_model_config_for_conversion(
    model_name: str, model_flavor: str
) -> Configurable.Config:
    """Build the unparallelized model config used for checkpoint conversion."""
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    return model_module.build_model_config(model_flavor)
