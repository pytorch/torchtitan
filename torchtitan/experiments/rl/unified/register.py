# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.rl.unified.infra.parallelize import parallelize_qwen3
from torchtitan.protocols.train_spec import TrainSpec
from vllm.logger import init_logger

from .vllm_wrapper import TorchTitanVLLMModelWrapper

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
        model_flavor: Model flavor key (e.g., "0.6B") to select from qwen3_args

    """
    from vllm.model_executor.models.registry import ModelRegistry

    # Get model_args directly from TrainSpec.model_args dict using flavor key
    if isinstance(train_spec.model_args, dict):
        if model_flavor not in train_spec.model_args:
            raise ValueError(
                f"Model flavor '{model_flavor}' not found in train_spec.model_args. "
                f"Available flavors: {list(train_spec.model_args.keys())}"
            )
        model_args = train_spec.model_args[model_flavor]
    else:
        raise ValueError(
            "train_spec.model_args must be a dict mapping flavor names to ModelArgs"
        )

    # Create dynamic model class directly from TrainSpec components
    class TorchTitanVLLMModelFromSpec(TorchTitanVLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_cls=train_spec.model_cls,
                model_args=model_args,
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
        f"(model_cls={train_spec.model_cls.__name__}, flavor={model_flavor})"
    )
