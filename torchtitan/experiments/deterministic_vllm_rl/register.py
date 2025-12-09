# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def register():
    """
    Register TorchTitan models with vLLM using class inheritance pattern.

    This function registers TorchTitan-trained models with vLLM's model registry
    by creating subclasses of TorchTitanVLLMWrapper and passing in the 5 core
    model-specific components:

    1. model_cls - The TorchTitan model class (e.g., Qwen3Model, Transformer)
    2. model_args_cls - The model args class (e.g., Qwen3ModelArgs)
    3. state_dict_adapter - State dict adapter for loading HF weights
    4. parallelize_fn - Function to apply tensor parallelism
    5. rope_cache_extension_fn - Optional function to extend RoPE cache

    """
    from vllm.model_executor.models.registry import ModelRegistry

    from torchtitan.experiments.deterministic_vllm_rl.models.base_wrapper import (
        TorchTitanVLLMWrapper,
    )

    from torchtitan.models.qwen3 import Qwen3Model, Qwen3ModelArgs
    from torchtitan.models.qwen3.infra.parallelize import (
        apply_non_moe_tp as apply_qwen3_tp,
    )
    from torchtitan.models.qwen3.model.model import precompute_rope_cache
    from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter

    class Qwen3TorchTitanForCausalLM(TorchTitanVLLMWrapper):
        """
        vLLM wrapper for TorchTitan-trained Qwen3 models.

        This class plugs in the 5 Qwen3-specific components into the
        generic TorchTitanVLLMWrapper.
        """

        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_cls=Qwen3Model,
                model_args_cls=Qwen3ModelArgs,
                state_dict_adapter=Qwen3StateDictAdapter,
                parallelize_fn=apply_qwen3_tp,
                rope_cache_compute_fn=precompute_rope_cache,
                vllm_config=vllm_config,
                prefix=prefix,
            )

    ModelRegistry.register_model(
        "Qwen3TorchTitanForCausalLM", Qwen3TorchTitanForCausalLM
    )
