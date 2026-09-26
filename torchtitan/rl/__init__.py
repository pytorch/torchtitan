# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To register TorchTitan models with vLLM:
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.rl.model.vllm_registry import register_to_vllm

    # Standalone inference (loads HF weights):
    register_to_vllm(
        model_config,
        parallelism=parallelism_config,
        compile_config=compile_config,
        checkpointer_config=CheckpointManager.Config(
            initial_load_in_hf=True,
            initial_load_path="/path/to/hf/checkpoint",
        ),
    )

    # RL loop (skip HF loading, weights from TorchStore):
    register_to_vllm(
        model_config,
        parallelism=parallelism_config,
        compile_config=compile_config,
        checkpointer_config=None,
    )
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torchtitan.rl._runtime import apply_env_defaults


# ``python -m torchtitan.rl.train`` executes this package initializer before
# train.py. Apply the defaults before a caller resolves either lazy vLLM API.
apply_env_defaults()

if TYPE_CHECKING:
    from torchtitan.rl.model.vllm_registry import register_to_vllm
    from torchtitan.rl.model.vllm_wrapper import VLLMModelWrapper


__all__ = [
    "VLLMModelWrapper",
    "register_to_vllm",  # Export register function for manual use
]


def __getattr__(name: str) -> Any:
    """Lazily import vLLM-backed public APIs.

    Most ``torchtitan.rl`` modules do not depend on vLLM. Keeping these
    re-exports lazy allows those modules to be imported in CPU-only
    environments while preserving the package-level public API.
    """
    if name == "register_to_vllm":
        from torchtitan.rl.model.vllm_registry import register_to_vllm

        return register_to_vllm
    if name == "VLLMModelWrapper":
        from torchtitan.rl.model.vllm_wrapper import VLLMModelWrapper

        return VLLMModelWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
