# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM-Compatible approach for deterministic RL training.

This module provides models that match vLLM's weight format (e.g., merged gate_up_proj)
with custom backward passes for gradient computation during training.
"""

from .batch_invariant_backward import (
    enable_batch_invariant_backward_mode,
    rms_norm_with_gradients,
    silu_and_mul_with_gradients,
)
from .models.attention import VLLMCompatibleFlashAttention
from .models.qwen3 import Qwen3VLLMCompatModel


__all__ = [
    "VLLMCompatibleFlashAttention",
    "Qwen3VLLMCompatModel",
    "enable_batch_invariant_backward_mode",
    "rms_norm_with_gradients",
    "silu_and_mul_with_gradients",
]
