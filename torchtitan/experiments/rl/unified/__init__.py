# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To manually load the plugin:
    from torchtitan.experiments.rl.unified import plugin
    plugin.register()
"""

from torchtitan.experiments.rl.unified.infra.parallelism_utils import (
    create_parallel_dims_from_vllm_config,
)
from torchtitan.experiments.rl.unified.models.vllm_wrapper import (
    TorchTitanVLLMModelWrapper,
)

# Export plugin register function for manual use (no auto-registration)
from torchtitan.experiments.rl.unified.plugin import register


__all__ = [
    "TorchTitanVLLMModelWrapper",
    "create_parallel_dims_from_vllm_config",
    "register",  # Export register function for manual use
]
