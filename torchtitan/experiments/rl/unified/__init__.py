# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To register TorchTitan models with vLLM:
    from torchtitan.experiments.rl.unified.plugin import register
    register(model_spec)
"""

from torchtitan.experiments.rl.unified.models.vllm_wrapper import (
    TorchTitanVLLMModelWrapper,
)

# Export plugin register function for manual use (no auto-registration)
from torchtitan.experiments.rl.unified.plugin import (
    register_model_to_vllm_model_registry,
)


__all__ = [
    "TorchTitanVLLMModelWrapper",
    "register_model_to_vllm_model_registry",  # Export register function for manual use
]
