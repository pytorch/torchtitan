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
from torchtitan.protocols.train_spec import get_train_spec, TrainSpec
from vllm.logger import init_logger

from .infra.parallelism_utils import create_parallel_dims_from_vllm_config

from .models.vllm_wrapper import TorchTitanVLLMModelWrapper

logger = init_logger(__name__)


def register_torchtitan_model_from_train_spec(
    train_spec: TrainSpec,
    model_name: str,
    model_flavor: str,
) -> None:
    """
    Register a TorchTitan model with vLLM using a TrainSpec.

    Args:
        train_spec: TorchTitan TrainSpec containing model components
        model_name: Name to register in vLLM (e.g., "Qwen3TorchTitanForCausalLM")
        model_flavor: Model flavor key (e.g., "0.6B") to select from qwen3_configs

    """
    from vllm.model_executor.models.registry import ModelRegistry

    # Get model_args directly from TrainSpec.model_configs dict using flavor key
    if isinstance(train_spec.model_configs, dict):
        if model_flavor not in train_spec.model_configs:
            raise ValueError(
                f"Model flavor '{model_flavor}' not found in train_spec.model_configs. "
                f"Available flavors: {list(train_spec.model_configs.keys())}"
            )
        model_config = train_spec.model_configs[model_flavor]
    else:
        raise ValueError(
            "train_spec.model_configs must be a dict mapping flavor names to Config"
        )

    # Create dynamic model class directly from TrainSpec components
    class TorchTitanVLLMModelFromSpec(TorchTitanVLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_config=model_config,
                state_dict_adapter=train_spec.state_dict_adapter,
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
        f"Successfully registered {model_name} with vLLM using TrainSpec "
        f"(flavor={model_flavor})"
    )


# Auto-register TorchTitan models with vLLM when this module is imported
register_torchtitan_model_from_train_spec(
    train_spec=get_train_spec("qwen3"),
    model_name="Qwen3TorchTitanForCausalLM",
    # TODO: Remove the model_flavor args when registering model,
    # allow passing model flavor option from config system. Now we have to specify
    # model_flavor during registration because we can not pass torchtitan job_config from LLM() Api
    model_flavor="0.6B",
)


__all__ = [
    "TorchTitanVLLMModelWrapper",
    "create_parallel_dims_from_vllm_config",
    "register_torchtitan_model_from_train_spec",
]
