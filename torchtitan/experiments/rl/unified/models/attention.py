# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torch.distributed.tensor import DTensor
from torchtitan.experiments.rl.vllm_compat.models.attention import (
    VLLMCompatibleFlashAttention,
)
from vllm.model_executor.layers.attention import Attention

logger = logging.getLogger(__name__)


class VLLMAttention(torch.nn.Module):
    """Adapter from TorchTitan tensor layout to ``vllm.Attention``.

    vLLM's ``Attention`` layer manages KV-cache and paged attention internally,
    but expects flattened ``(num_tokens, num_heads, head_dim)`` inputs.  This
    wrapper handles the transpose/reshape from TorchTitan's
    ``(batch, num_heads, seq_len, head_dim)`` layout and back.

    Used by the **generator** (via :func:`replace_with_vllm_attention`).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_name: str,
        scale: float | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_name = layer_name

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        if scale is None:
            self.scale = head_dim**-0.5
        else:
            self.scale = scale

        cache_config = (
            vllm_config.cache_config if hasattr(vllm_config, "cache_config") else None
        )

        self.vllm_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=self.scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"model.layers.{layer_name}.attention.inner_attention",
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """Run vLLM paged attention.

        Args:
            q: ``(batch, num_heads, seq_len, head_dim)``
            k: ``(batch, num_kv_heads, seq_len, head_dim)``
            v: ``(batch, num_kv_heads, seq_len, head_dim)``
            scale: Ignored — vLLM uses its own internal scale.
            enable_gqa: Ignored — vLLM handles GQA internally.

        Returns:
            ``(batch, num_heads, seq_len, head_dim)``
        """
        # Unwrap DTensor inputs to local tensors for attention computation
        device_mesh = None
        placements = None
        if isinstance(q, DTensor):
            device_mesh = q.device_mesh
            placements = q.placements
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

        # Input is (batch, num_heads, seq_len, head_dim)
        # TODO: may be good to use einops in future as we can explicitly reshape
        # with dimension names - see https://github.com/arogozhnikov/einops
        batch_size, num_heads, seq_len, head_dim = q.shape
        _, num_kv_heads, _, _ = k.shape

        # Transpose to (batch, seq_len, num_heads, head_dim) for vLLM
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # TODO: reimplement as a 4d tensor once vLLM fix has landed
        # Then flatten batch and seq_len: (batch * seq_len, num_heads, head_dim)
        q = q.reshape(batch_size * seq_len, num_heads, head_dim)
        k = k.reshape(batch_size * seq_len, num_kv_heads, head_dim)
        v = v.reshape(batch_size * seq_len, num_kv_heads, head_dim)

        # vLLM attention returns (num_tokens, hidden_size) where hidden_size = num_heads * head_dim
        output_flat = self.vllm_attn(q, k, v)

        # Output is (batch * seq_len, num_heads * head_dim), reshape to (batch, seq_len, num_heads, head_dim)
        output = output_flat.view(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        # Wrap output back as DTensor if inputs were DTensors
        if device_mesh is not None:
            output = DTensor.from_local(
                output, device_mesh=device_mesh, placements=placements
            )

        return output


def replace_with_vllm_attention(model, tp_degree=1):
    """Replace ``inner_attention`` with :class:`VLLMAttention`.

    **Generator side.** Used by ``TorchTitanVLLMModelWrapper`` because:

    1. ``vllm.Attention`` manages KV-cache and paged attention for inference.
    2. Head counts are divided by *tp_degree* so each TP rank holds the
       correct shard of Q / KV heads.

    Args:
        model: TorchTitan model with ``.layers`` and ``.config``.
        tp_degree: Tensor-parallel world size.
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

    logger.info(
        f"Successfully replaced TorchTitan attention with VLLMAttention "
        f"({len(model.layers)} layers)"
    )


def replace_with_vllm_compatible_flash_attention(model, tp_size=1):
    """Replace ``inner_attention`` with :class:`VLLMCompatibleFlashAttention`.

    **Trainer side.** Called on the ``PolicyTrainer`` model because:

    1. The generator's ``vllm.Attention`` (see :func:`replace_with_vllm_attention`)
       uses vLLM's flash-attention kernel internally.  To achieve **bitwise
       identical** forward outputs between trainer and generator, we patch the
       trainer's attention to the same flash-attention kernel.
    2. Training requires gradients.  ``VLLMCompatibleFlashAttention`` wraps
       vLLM's flash-attention kernel with a custom backward pass so gradients
       can flow during RL policy updates.

    Args:
        model: TorchTitan model with ``.layers`` and ``.config``.
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
