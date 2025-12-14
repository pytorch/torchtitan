# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dion optimizer integration for TorchTitan.

This package provides the Dion optimizer implementation and its integration
with TorchTitan's training framework.
"""

from .dion import Dion, DionMixedPrecisionConfig, DionParamConfig

__all__ = [
    "Dion",
    "DionMixedPrecisionConfig",
    "DionParamConfig",
]
