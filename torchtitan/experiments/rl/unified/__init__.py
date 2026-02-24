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
    create_torchtitan_config_from_vllm,
    TorchTitanVLLMModelWrapper,
)
from torchtitan.experiments.rl.unified.plugin import register


__all__ = [
    "TorchTitanVLLMModelWrapper",
    "create_torchtitan_config_from_vllm",
    "register",  # Export register function for manual use
]
