# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 model with vLLM compatibility for deterministic RL training.
"""

from .model_batch_invariant import Qwen3VLLMCompatModel
from .model_vllm_compat import TorchTitanQwen3ForCausalLM

__all__ = ["Qwen3VLLMCompatModel", "TorchTitanQwen3ForCausalLM"]
