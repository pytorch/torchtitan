# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flavors import build_model_config, MODEL_FLAVORS
from .model import DeepSeekV4Model

__all__ = ["DeepSeekV4Model", "MODEL_FLAVORS", "build_model_config"]
