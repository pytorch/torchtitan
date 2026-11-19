# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.kimi_k3.model import KimiK3Model, KimiK3MultimodalModel

from .flavors import build_model_config, MODEL_FLAVORS

__all__ = [
    "KimiK3Model",
    "KimiK3MultimodalModel",
    "MODEL_FLAVORS",
    "build_model_config",
]
