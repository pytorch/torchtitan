# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

from torchtitan.experiments.rl.unified.models.attention import VLLMAttention
from torchtitan.experiments.rl.vllm_compat.models.attention import (
    VLLMCompatibleFlashAttention,
)

logger = logging.getLogger(__name__)


def replace_with_vllm_attention(model, tp_degree=1):
    """
    Replace TorchTitan attention with vLLM's Attention.

    Assumes model has .layers dict with .attention.inner_attention structure.
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"Model {type(model).__name__} must have .layers attribute"
        )

    model_args = model.config

    # Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py#L80
    # Calculate num_kv_heads based on TP size
    total_num_kv_heads = model_args.layer.attention.n_kv_heads
    if total_num_kv_heads >= tp_degree:
        # Number of KV heads is greater than TP size, so we partition
        # the KV heads across multiple tensor parallel GPUs.
        assert total_num_kv_heads % tp_degree == 0
        num_kv_heads = total_num_kv_heads // tp_degree
    else:
        # TODO: Handle this branch correctly
        raise ValueError("num_kv_heads are smaller than tp_degree")

    for layer_name, layer in model.layers.items():
        if not hasattr(layer, "attention"):
            raise ValueError(f"Layer {layer_name} must have .attention attribute")

        # GQA
        head_dim = model_args.layer.attention.head_dim
        vllm_attn = VLLMAttention(
            hidden_size=model_args.dim,
            num_heads=model_args.layer.attention.n_heads // tp_degree,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_name=layer_name,
            scale=head_dim**-0.5,
        )

        layer.attention.inner_attention = vllm_attn

    logger.debug(
        f"Successfully replaced TorchTitan attention with VLLMAttention "
        f"({len(model.layers)} layers)"
    )


def replace_with_vllm_compatible_flash_attention(model):
    """
    Replace TorchTitan attention with vLLM compatible flash attention.

    Assumes model has .layers dict with .attention.inner_attention structure.
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"Model {type(model).__name__} must have .layers attribute"
        )

    for layer_name, layer in model.layers.items():
        if not hasattr(layer, "attention"):
            raise ValueError(f"Layer {layer_name} must have .attention attribute")

        vllm_attn = VLLMCompatibleFlashAttention()

        layer.attention.inner_attention = vllm_attn

    logger.debug(
        f"Successfully replaced TorchTitan attention with VLLMCompatibleFlashAttention "
        f"({len(model.layers)} layers)"
    )
