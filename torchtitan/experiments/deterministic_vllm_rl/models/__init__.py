# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Models for deterministic vLLM RL training.

This module automatically registers TorchTitan models with vLLM when imported.
"""

from vllm.logger import init_logger

from torchtitan.protocols.train_spec import get_train_spec, TrainSpec
from .attention import VLLMCompatibleFlashAttention, VLLMPagedFlashAttention
from .base_wrapper import TorchTitanVLLMModel


logger = init_logger(__name__)


def register_torchtitan_model_from_train_spec(
    train_spec: TrainSpec,
    model_name: str,
) -> None:
    """
    Register a TorchTitan model with vLLM using a TrainSpec.

    Args:
        train_spec: TorchTitan TrainSpec containing model components
        model_name: Name to register in vLLM (e.g., "Qwen3TorchTitanForCausalLM")

    """
    from vllm.model_executor.models.registry import ModelRegistry

    # Extract model_args from TrainSpec
    # TrainSpec has model_args as a Mapping, get the first value
    if isinstance(train_spec.model_args, dict):
        model_args_cls = type(next(iter(train_spec.model_args.values())))
    else:
        model_args_cls = train_spec.model_args

    # Create dynamic model class directly from TrainSpec components
    class TorchTitanVLLMModelFromSpec(TorchTitanVLLMModel):
        """Dynamically created vLLM model from TrainSpec."""

        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_cls=train_spec.model_cls,
                model_args_cls=model_args_cls,
                state_dict_adapter=train_spec.state_dict_adapter,
                parallelize_fn=train_spec.parallelize_fn,
                vllm_config=vllm_config,
                prefix=prefix,
            )

    # Set the class name
    TorchTitanVLLMModelFromSpec.__name__ = model_name
    TorchTitanVLLMModelFromSpec.__qualname__ = model_name

    # Register with vLLM
    ModelRegistry.register_model(model_name, TorchTitanVLLMModelFromSpec)


# Auto-register TorchTitan models with vLLM when this module is imported
# NOTE: We use a custom parallelization function for vLLM compatibility
from torchtitan.protocols.train_spec import TrainSpec

register_torchtitan_model_from_train_spec(
    train_spec=get_train_spec("qwen3"),
    model_name="Qwen3TorchTitanForCausalLM",
)


__all__ = [
    "VLLMCompatibleFlashAttention",
    "VLLMPagedFlashAttention",
    "TorchTitanVLLMModel",
]
