# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic RL training with vLLM experiment.

This experiment provides tools for bitwise-deterministic reinforcement learning
training using vLLM for fast rollouts and TorchTitan for training.

Key components:
- VLLMCompatibleFlashAttention: Flash attention with custom backward pass
- Qwen3VLLMCompatModel: vLLM-compatible model with merged projections
- batch_invariant_backward: Gradient support for vLLM's deterministic operations
- simple_rl: End-to-end RL training loop
"""

from .batch_invariant_backward import (
    enable_batch_invariant_backward_mode,
    rms_norm_with_gradients,
    silu_and_mul_with_gradients,
)
from .models import VLLMCompatibleFlashAttention
from .models.qwen3 import Qwen3VLLMCompatModel

__all__ = [
    "VLLMCompatibleFlashAttention",
    "Qwen3VLLMCompatModel",
    "enable_batch_invariant_backward_mode",
    "rms_norm_with_gradients",
    "silu_and_mul_with_gradients",
]
