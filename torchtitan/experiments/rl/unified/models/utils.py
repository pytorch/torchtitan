# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from enum import Enum

import torch
from safetensors.torch import load_file
from torchtitan.experiments.rl.unified.models.attention import VLLMAttention
from torchtitan.experiments.rl.vllm_compat.models.attention import (
    VLLMCompatibleFlashAttention,
)
from torchtitan.experiments.rl.vllm_compat.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)
from torchtitan.models.qwen3.model import Qwen3Model
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class ModelMode(str, Enum):
    """
    Enum defining which TorchTitan model to use.

    Attributes:
        UNIFIED: Standard TorchTitan model replaced with vLLM attention for unified
            training and inference.
        VLLM_COMPAT: vLLM-compatible TorchTitan model using vLLM's batch invariant kernels,
            ensuring bitwise determinism between training and inference.
        STANDARD: Plain TorchTitan model without any modifications.
    """

    UNIFIED = "unified"
    VLLM_COMPAT = "vllm_compat"
    STANDARD = "standard"


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
        vllm_attn = VLLMAttention(
            hidden_size=model_args.dim,
            num_heads=model_args.layer.attention.n_heads // tp_degree,
            num_kv_heads=num_kv_heads,
            head_dim=model_args.head_dim,
            layer_name=layer_name,
            scale=model_args.head_dim**-0.5,
        )

        layer.attention.inner_attention = vllm_attn

    logger.info(
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

    logger.info(
        f"Successfully replaced TorchTitan attention with VLLMCompatibleFlashAttention "
        f"({len(model.layers)} layers)"
    )


def load_model(
    checkpoint_path: str, model_path: str, model_mode: str = ModelMode.VLLM_COMPAT
):
    """
    Load TorchTitan model from checkpoint for trainer.

    Args:
        checkpoint_path: Path to TorchTitan checkpoint
        model_path: Path to HuggingFace model (for config)
        model_mode: Indicates which model to use. Train inferece unified model, batch invariant Torchtitan model,
            or plain Torchtitan model

    Returns:
        model: Loaded TorchTitan model for trainer.
    """
    # Load HuggingFace config
    # TODO: do not depend on transformers.AutoConfig, use qwen_args directly
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Create model args
    model_args = Qwen3Model.Config(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        hidden_dim=hf_config.intermediate_size,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
        qk_norm=True,
        depth_init=True,
        eos_id=getattr(hf_config, "eos_token_id", 151645),
        enable_weight_tying=getattr(hf_config, "tie_word_embeddings", False),
    )

    # state_dict is in standard TorchTitan format (w1, w2, w3)
    state_dict = load_file(checkpoint_path)

    # If weight tying is enabled but output.weight is missing from the checkpoint
    # (HF models with tie_word_embeddings=True may not store lm_head.weight),
    # synthesize it from tok_embeddings.weight so load_state_dict(strict=True) works.
    if model_args.enable_weight_tying and "output.weight" not in state_dict:
        state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

    if model_mode == ModelMode.UNIFIED:
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args)
        # Set global default dtype to bfloat16. This is needed because vLLM's Attention
        # layer uses torch.get_default_dtype() and it doesn't support float32
        torch.set_default_dtype(torch.bfloat16)
        # NOTE: Override attention to vllm compatible attention for backward capability.
        # Only patch to vllm compatible attention for training.
        replace_with_vllm_compatible_flash_attention(model)

        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=True)
    elif model_mode == ModelMode.VLLM_COMPAT:
        # Create and load model that has bitwise determinism between training and inference
        from torchtitan.experiments.rl.vllm_compat.models.qwen3 import (
            Qwen3VLLMCompatModel,
        )

        model = Qwen3VLLMCompatModel(model_args)
        # Convert to vLLM-compat format (merged gate_up_proj, down_proj)
        vllm_compat_state = torchtitan_to_vllm_compat(state_dict)
        model.load_state_dict(vllm_compat_state, strict=False)
    else:
        # Use standard TorchTitan model
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args)
        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=False)

    model.to(torch.bfloat16)

    return model
