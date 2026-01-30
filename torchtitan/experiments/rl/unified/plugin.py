# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM plugin for TorchTitan models.
"""


def register(model_flavor: str):
    """
    Register TorchTitan models with vLLM.

    This function registers TorchTitan models (e.g., Qwen3) with vLLM's ModelRegistry.
    It's called automatically when the unified module is imported, or can be called
    explicitly for manual registration.
    """
    # NOTE: This is a temporary parallel plan patch for Qwen3 model. Will be replace by parallel plan in core torchtitan.
    from torchtitan.experiments.rl.unified.infra.parallelize import parallelize_qwen3
    from torchtitan.experiments.rl.unified.models.vllm_wrapper import (
        TorchTitanVLLMModelWrapper,
    )
    from torchtitan.protocols.train_spec import get_train_spec, TrainSpec
    from vllm.logger import init_logger
    from vllm.model_executor.models.registry import ModelRegistry

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

        # Register with vLLM's ModelRegistry
        ModelRegistry.register_model(model_name, TorchTitanVLLMModelFromSpec)

        logger.info(
            f"Successfully registered {model_name} with vLLM using TrainSpec "
            f"(model_cls={train_spec.model_cls.__name__}, flavor={model_flavor})"
        )

    # Register TorchTitan models
    # TODO: Make this configurable instead of hardcoded
    register_torchtitan_model_from_train_spec(
        train_spec=get_train_spec("qwen3"),
        model_name="Qwen3TorchTitanForCausalLM",
        # TODO: Remove the model_flavor args when registering model,
        # allow passing model flavor option from config system. Now we have to specify
        # model_flavor during registration because we can not pass torchtitan job_config from LLM() Api
        model_flavor=model_flavor,
    )
