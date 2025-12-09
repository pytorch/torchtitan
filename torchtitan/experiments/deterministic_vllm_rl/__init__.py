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
- TorchTitanVLLMModel: Generic wrapper for TorchTitan models with vLLM

For vLLM inference with TorchTitan models, see:
- models/base_wrapper.py: Core vLLM wrapper
- models/__init__.py: Auto-registration with vLLM
- infer.py: Example inference script
"""

from .batch_invariant_backward import (
    enable_batch_invariant_backward_mode,
    rms_norm_with_gradients,
    silu_and_mul_with_gradients,
)
from .models import VLLMCompatibleFlashAttention
from .models.base_wrapper import TorchTitanVLLMModel
from .models.qwen3 import Qwen3VLLMCompatModel


__all__ = [
    "VLLMCompatibleFlashAttention",
    "Qwen3VLLMCompatModel",
    "enable_batch_invariant_backward_mode",
    "rms_norm_with_gradients",
    "silu_and_mul_with_gradients",
    "TorchTitanVLLMModel",
]
