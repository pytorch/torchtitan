# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def register():
    """
    Register TorchTitan models with vLLM.

    This function is called to register TorchTitan-trained models with vLLM.
    It sets up the necessary model registry entries for TorchTitan models.

    Currently supports:
    - Qwen3TorchTitanForCausalLM: Qwen3 models trained with TorchTitan

    """
    from vllm.model_executor.models.registry import ModelRegistry

    from torchtitan.experiments.deterministic_vllm_rl.models.qwen3 import (
        TorchTitanQwen3ForCausalLM,
    )

    # Register Qwen3TorchTitanForCausalLM with vLLM's ModelRegistry
    # This maps the architecture name from config.json to the model class
    ModelRegistry.register_model(
        "Qwen3TorchTitanForCausalLM", TorchTitanQwen3ForCausalLM
    )
