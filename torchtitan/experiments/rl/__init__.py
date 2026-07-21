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

import os
import sys
import warnings

# Avoid memory fragmentation and peak reserved memory increasing over time
# To overwrite, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    if "torch" in sys.modules:
        warnings.warn(
            "The 'torch' module has already been imported. "
            "Setting PYTORCH_CUDA_ALLOC_CONF may not have an effect."
            "For best results, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before importing 'torch'."
        )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

__all__ = [
    "VLLMModelWrapper",
    "register_to_vllm",  # Export register function for manual use
]


# Import lazily (PEP 562): eagerly importing vllm_wrapper here pulls the vLLM
# attention/GDN cores, whose @eager_break_during_capture decorators read
# VLLM_USE_BREAKABLE_CUDAGRAPH at import time. Deferring the import lets callers
# set that env (based on the config's cudagraph mode) before register_to_vllm
# triggers it, so breakable cudagraph is honored for FULL_AND_PIECEWISE.
def __getattr__(name: str):
    if name == "register_to_vllm":
        from torchtitan.experiments.rl.models.vllm_registry import register_to_vllm

        return register_to_vllm
    if name == "VLLMModelWrapper":
        from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper

        return VLLMModelWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
