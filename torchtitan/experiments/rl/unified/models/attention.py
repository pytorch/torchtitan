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
from torchtitan.protocols.module import Module
from vllm.model_executor.layers.attention import Attention

logger = logging.getLogger(__name__)


class VLLMAttention(Module):
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
        # Capture the original symbolic seq_len from the input BEFORE
        # to_local() so that the symbol is the same one GQAttention uses
        # in its .view(bs, seqlen, -1) call.
        batch_size, _, seq_len, head_dim = q.shape

        # Unwrap DTensor inputs to local tensors for attention computation
        device_mesh = None
        placements = None
        if isinstance(q, DTensor):
            device_mesh = q.device_mesh
            placements = q.placements
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

        # TODO: may be good to use einops in future as we can explicitly reshape
        # with dimension names - see https://github.com/arogozhnikov/einops
        # Convert from (batch, num_heads, seq_len, head_dim)
        #   to (batch*seq_len, num_heads (or num_kv_heads), head_dim) for vLLM Attn
        q = q.transpose(1, 2).reshape(batch_size * seq_len, -1, head_dim)
        k = k.transpose(1, 2).reshape(batch_size * seq_len, -1, head_dim)
        v = v.transpose(1, 2).reshape(batch_size * seq_len, -1, head_dim)

        # vLLM attention returns (num_tokens, num_heads/num_kv_heads * head_dim)
        output_flat = self.vllm_attn(q, k, v)

        # vLLM's flash attention backend may pad the token count (e.g.
        # round up to an even number), which introduces a new symbolic
        # shape under torch.compile.  Narrow to trim this padding
        # NOTE: this error only happens when batch_size and seq_len are 1
        # which happens with cudagraph capture for dummy input
        output_flat = output_flat.narrow(0, 0, batch_size * seq_len)

        # Reshape back to titan: (batch, num_heads_local, seq_len, head_dim)
        output = output_flat.view(batch_size, seq_len, -1, head_dim)
        output = output.transpose(1, 2)

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
