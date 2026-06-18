# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To register TorchTitan models with vLLM:
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.experiments.rl.models.vllm_registry import register_to_vllm

    # Standalone inference (loads HF weights):
    register_to_vllm(
        model_spec,
        parallelism=parallelism_config,
        compile_config=compile_config,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path="/path/to/hf/checkpoint",
        ),
    )

    # RL loop (skip HF loading, weights from TorchStore):
    register_to_vllm(
        model_spec,
        parallelism=parallelism_config,
        compile_config=compile_config,
        checkpoint_config=CheckpointManager.Config(enable=False),
    )
"""

from torchtitan.experiments.rl.models.vllm_registry import register_to_vllm
from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper


__all__ = [
    "VLLMModelWrapper",
    "register_to_vllm",  # Export register function for manual use
]
