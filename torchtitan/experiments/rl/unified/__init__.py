# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

This module automatically registers TorchTitan models with vLLM when imported.
Uses the canonical TorchTitan model definition directly with vLLM inference engine.
"""

from torchtitan.experiments.rl.unified.infra.parallelize import parallelize_qwen3
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
from torchtitan.protocols.model_spec import ModelSpec
from vllm.logger import init_logger

from .infra.parallelism_utils import create_parallel_dims_from_vllm_config

from .models.vllm_wrapper import TorchTitanVLLMModelWrapper

logger = init_logger(__name__)


def register_torchtitan_model_from_model_spec(
    model_spec: ModelSpec,
    model_name: str,
) -> None:
    """
    Register a TorchTitan model with vLLM using a ModelSpec.

    Args:
        model_spec: TorchTitan ModelSpec containing model components
        model_name: Name to register in vLLM (e.g., "Qwen3TorchTitanForCausalLM")

    """
    from vllm.model_executor.models.registry import ModelRegistry

    model_config = model_spec.model

    # Create dynamic model class directly from ModelSpec components
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

    # Set the class name
    TorchTitanVLLMModelFromSpec.__name__ = model_name
    TorchTitanVLLMModelFromSpec.__qualname__ = model_name

    # Register with vLLM
    ModelRegistry.register_model(model_name, TorchTitanVLLMModelFromSpec)

    logger.info(
        f"Successfully registered {model_name} with vLLM using ModelSpec "
        f"(flavor={model_spec.flavor})"
    )


# Auto-register TorchTitan models with vLLM when this module is imported
register_torchtitan_model_from_model_spec(
    model_spec=qwen3_model_registry("0.6B"),
    model_name="Qwen3TorchTitanForCausalLM",
)


__all__ = [
    "TorchTitanVLLMModelWrapper",
    "create_parallel_dims_from_vllm_config",
    "register_torchtitan_model_from_model_spec",
]
