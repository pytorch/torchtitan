# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flavors import build_model_config, MINIMAX_M3_SPECIAL_TOKENS, MODEL_FLAVORS
from .model import MiniMaxM3Model

__all__ = [
    "MiniMaxM3Model",
    "MINIMAX_M3_SPECIAL_TOKENS",
    "MODEL_FLAVORS",
    "build_model_config",
]
