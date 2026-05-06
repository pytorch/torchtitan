# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To register TorchTitan models with vLLM:
    from torchtitan.experiments.rl.models.vllm_registry import registry_to_vllm
    registry_to_vllm(model_spec)
"""

from torchtitan.experiments.rl.models.vllm_registry import registry_to_vllm
from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper


__all__ = [
    "TorchTitanVLLMModelWrapper",
    "registry_to_vllm",  # Export register function for manual use
]
