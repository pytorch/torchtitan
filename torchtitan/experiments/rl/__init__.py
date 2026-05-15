# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To register TorchTitan models with vLLM:
    from torchtitan.experiments.rl.models.vllm_registry import registry_to_vllm
    registry_to_vllm(
        model_spec,
        parallelism=parallelism_config,
        compile_config=compile_config,
    )
"""

__all__ = [
    "VLLMModelWrapper",
    "registry_to_vllm",
]


# Lazy imports: defer loading GPU/vLLM dependencies until actually used.
# This allows CPU-only unit tests to import RL utilities without failing.
def __getattr__(name):
    if name == "VLLMModelWrapper":
        from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper

        return VLLMModelWrapper
    elif name == "registry_to_vllm":
        from torchtitan.experiments.rl.models.vllm_registry import registry_to_vllm

        return registry_to_vllm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
