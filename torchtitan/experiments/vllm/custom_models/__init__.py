# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for integrating custom model implementations with vLLM."""

from .custom_model_wrapper import VLLMModelForCausalLM
from .utils import store_positions_in_context

__all__ = [
    # Base wrapper
    "VLLMModelForCausalLM",
    # Utilities
    "store_positions_in_context",
]
