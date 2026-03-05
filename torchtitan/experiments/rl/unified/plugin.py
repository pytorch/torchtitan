# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM plugin for TorchTitan models.

Usage:
    from torchtitan.experiments.rl.unified.plugin import register_model_to_vllm_model_registry
    register_model_to_vllm_model_registry(model_spec)
"""

from torchtitan.protocols.model_spec import ModelSpec

# Model-agnostic name used for vLLM model registration.
# Must match the hf_overrides["architectures"] value passed to EngineArgs.
VLLM_MODEL_NAME = "TorchTitanCausalLM"


def register_model_to_vllm_model_registry(
    model_spec: ModelSpec,
) -> None:
    """
    Register a TorchTitan model with vLLM's ModelRegistry.

    Must be called before creating a vLLM engine that uses this model.

    Args:
        model_spec: TorchTitan ModelSpec containing model config and components
    """
    from torchtitan.experiments.rl.unified.models.vllm_wrapper import (
        TorchTitanVLLMModelWrapper,
    )
    from vllm.logger import init_logger
    from vllm.model_executor.models.registry import ModelRegistry

    logger = init_logger(__name__)

    # Create dynamic model class capturing ModelSpec in the closure
    class TorchTitanVLLMModelFromSpec(TorchTitanVLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_spec=model_spec,
                vllm_config=vllm_config,
                prefix=prefix,
            )

    # Set the class name so vLLM can identify it
    TorchTitanVLLMModelFromSpec.__name__ = VLLM_MODEL_NAME
    TorchTitanVLLMModelFromSpec.__qualname__ = VLLM_MODEL_NAME

    # Register with vLLM
    ModelRegistry.register_model(VLLM_MODEL_NAME, TorchTitanVLLMModelFromSpec)

    logger.info(
        f"Registered {VLLM_MODEL_NAME} with vLLM "
        f"(model={model_spec.name}, flavor={model_spec.flavor})"
    )
