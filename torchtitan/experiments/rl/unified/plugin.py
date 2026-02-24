# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM plugin for TorchTitan models.

Usage:
    from torchtitan.experiments.rl.unified.plugin import register
    register(model_spec)
"""

from torchtitan.protocols.model_spec import ModelSpec


def register(
    model_spec: ModelSpec,
    model_name: str = "Qwen3TorchTitanForCausalLM",
) -> None:
    """
    Register a TorchTitan model with vLLM's ModelRegistry.

    Must be called before creating a vLLM engine that uses this model.

    Args:
        model_spec: TorchTitan ModelSpec containing model config and components
        model_name: Name to register in vLLM (must match hf_overrides["architectures"])
    """
    from torchtitan.experiments.rl.unified.infra.parallelize import parallelize_qwen3
    from torchtitan.experiments.rl.unified.models.vllm_wrapper import (
        TorchTitanVLLMModelWrapper,
    )
    from vllm.logger import init_logger
    from vllm.model_executor.models.registry import ModelRegistry

    logger = init_logger(__name__)

    model_config = model_spec.model

    # Create dynamic model class capturing ModelSpec components in the closure
    class TorchTitanVLLMModelFromSpec(TorchTitanVLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_config=model_config,
                state_dict_adapter=model_spec.state_dict_adapter,
                # NOTE: This should be replaced with qwen3 parallelization plan in torchtitan core
                parallelize_fn=parallelize_qwen3,
                vllm_config=vllm_config,
                prefix=prefix,
            )

    # Set the class name so vLLM can identify it
    TorchTitanVLLMModelFromSpec.__name__ = model_name
    TorchTitanVLLMModelFromSpec.__qualname__ = model_name

    # Register with vLLM
    ModelRegistry.register_model(model_name, TorchTitanVLLMModelFromSpec)

    logger.info(
        f"Registered {model_name} with vLLM "
        f"(model={model_spec.name}, flavor={model_spec.flavor})"
    )
