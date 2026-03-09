# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import types

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding
from torchtitan.experiments.rl.vllm_compat.models.attention import (
    VLLMCompatibleFlashAttention,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Non-mutating rotary embedding to avoid auto-functionalize overhead.
#
# vLLM's ``_C::rotary_embedding`` mutates query and key in-place, which causes
# aot-eager to wrap it with an auto_functionalized_v2 wrapper (~400 us overhead
# per call).  We register a functional variant that clones query/key, applies
# the in-place kernel, and returns them.
# ---------------------------------------------------------------------------
def _rotary_embedding_return_tensors(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = query.clone()
    k_out = key.clone()
    torch.ops._C.rotary_embedding(positions, q_out, k_out, head_size, cos_sin_cache, is_neox)
    return q_out, k_out


def _rotary_embedding_return_tensors_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query.clone(), key.clone()


direct_register_custom_op(
    op_name="rotary_embedding_return_tensors",
    op_func=_rotary_embedding_return_tensors,
    mutates_args=[],
    fake_impl=_rotary_embedding_return_tensors_fake,
)

logger.info("Registered non-mutating rotary_embedding_return_tensors custom op")


# DTensor sharding strategy for the non-mutating rotary embedding op
@register_sharding(torch.ops.vllm.rotary_embedding_return_tensors.default)
def _rotary_embedding_return_tensors_sharding(positions, query, key, head_size, cos_sin_cache, is_neox):
    """Register DTensor sharding for the non-mutating rotary embedding op."""
    return [
        (
            [Shard(1), Shard(1)],  # outputs: query and key sharded on dim 1
            [Replicate(), Shard(1), Shard(1), None, Replicate(), None],
        ),
    ]


logger.info("Registered DTensor sharding strategy for rotary_embedding_return_tensors")


# ---------------------------------------------------------------------------
# Non-mutating attention custom op to avoid auto-functionalize overhead.
#
# vLLM's ``unified_attention_with_output`` mutates the output buffer in-place,
# which causes aot-eager to wrap it with an auto-functionalize wrapper (~300 us
# overhead per call).  Since torchtitan doesn't need the in-place semantics
# (it just takes the return value), we register a functional variant that
# allocates output internally and returns it.
# ---------------------------------------------------------------------------
def _unified_attention_return_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> torch.Tensor:
    del kv_cache_dummy_dep
    attn_metadata, attn_layer, kv_cache, _ = get_attention_context(layer_name)
    output = torch.empty_like(query)
    attn_layer.impl.forward(
        attn_layer, query, key, value, kv_cache, attn_metadata, output=output
    )
    return output


def _unified_attention_return_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


direct_register_custom_op(
    op_name="unified_attention_return_output",
    op_func=_unified_attention_return_output,
    mutates_args=[],
    fake_impl=_unified_attention_return_output_fake,
)

logger.info("Registered non-mutating unified_attention_return_output custom op")


def _torchtitan_to_vllm_cos_sin_cache(
    rope_cache: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Convert torchtitan's rope_cache to vLLM's cos_sin_cache format.

    Torchtitan: (max_seq_len, head_dim * 2) = [cos_0..cos_{d-1}, sin_0..sin_{d-1}]
        where cos/sin values are duplicated (first half == second half).
    vLLM:       (max_seq_len, head_dim) = [cos_0..cos_{d/2-1}, sin_0..sin_{d/2-1}]
    """
    half = head_dim // 2
    cos_part = rope_cache[:, :half]
    sin_part = rope_cache[:, head_dim : head_dim + half]
    return torch.cat([cos_part, sin_part], dim=-1)


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

        # Store the vLLM layer name for direct op calls
        self.vllm_layer_name = self.vllm_attn.layer_name
        self._needs_separate_kv_update = (
            not self.vllm_attn.attn_backend.forward_includes_kv_cache_update
        )

        self.layer_name = ''

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run vLLM paged attention.

        Args:
            q: ``(batch, seq_len, num_heads, head_dim)``
            k: ``(batch, seq_len, num_kv_heads, head_dim)``
            v: ``(batch, seq_len, num_kv_heads, head_dim)``

        Returns:
            ``(batch, seq_len, num_heads * head_dim)``
        """
        # Unwrap DTensor inputs to local tensors for attention computation
        device_mesh = None
        if isinstance(q, DTensor):
            device_mesh = q.device_mesh
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

        # Input is (batch, seq_len, num_heads, head_dim) — no transpose needed
        batch_size, seq_len, num_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k.shape

        # Flatten batch and seq_len: (batch * seq_len, num_heads, head_dim)
        q = q.reshape(batch_size * seq_len, num_heads, head_dim)
        k = k.reshape(batch_size * seq_len, num_kv_heads, head_dim)
        v = v.reshape(batch_size * seq_len, num_kv_heads, head_dim)

        # Bypass Attention.forward() and call ops directly to use the
        # non-mutating attention op (avoids auto-functionalize overhead).
        kv_cache_dummy_dep = None
        if self._needs_separate_kv_update:
            kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                k, v, self.vllm_layer_name
            )
        output_flat = torch.ops.vllm.unified_attention_return_output(
            q, k, v, self.vllm_layer_name, kv_cache_dummy_dep
        )

        # Output is (batch * seq_len, num_heads, head_dim),
        # reshape directly to (batch, seq_len, num_heads * head_dim)
        output = output_flat.view(batch_size, seq_len, -1)

        # Wrap output back as DTensor if inputs were DTensors
        if device_mesh is not None:
            output = DTensor.from_local(
                output, device_mesh=device_mesh, placements=[Shard(-1)]
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

    # Also replace the RoPE implementation with vLLM's fused CUDA kernel
    _replace_rope_with_vllm_kernel(model)

    logger.info(
        f"Successfully replaced TorchTitan attention with VLLMAttention "
        f"({len(model.layers)} layers)"
    )


def _replace_rope_with_vllm_kernel(model):
    """Replace torchtitan's broken-down RoPE with vLLM's fused CUDA kernel.

    Monkey-patches GQAttention.forward to call torch.ops._C.rotary_embedding
    instead of apply_rotary_emb_cos_sin. The DTensor sharding strategy is
    registered at module load time above, so the kernel works with DTensor
    inputs directly.
    """
    from torchtitan.models.common.attention import (
        apply_rotary_emb_complex,
        AttentionMasksType,
        GQAttention,
    )

    def _patched_forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        if self.use_rope:
            if self.rope_backend == "cos_sin":
                assert positions is not None, (
                    "vLLM RoPE kernel requires explicit positions"
                )
                # Convert cache format once and cache it
                if not hasattr(self, '_vllm_cos_sin_cache') or self._vllm_cos_sin_cache.shape[0] < rope_cache.shape[0]:
                    self._vllm_cos_sin_cache = _torchtitan_to_vllm_cos_sin_cache(
                        rope_cache, self.head_dim
                    )
                cos_sin_cache = self._vllm_cos_sin_cache.to(
                    dtype=xq.dtype, device=xq.device
                )

                # Flatten for vLLM kernel: (num_tokens, num_heads*head_dim)
                # DTensor sharding strategy handles unwrap/rewrap automatically
                num_tokens = bs * seqlen
                flat_q = xq.reshape(num_tokens, -1)
                flat_k = xk.reshape(num_tokens, -1)
                flat_pos = positions.reshape(num_tokens)

                # Non-mutating RoPE to avoid auto-functionalize overhead
                flat_q, flat_k = torch.ops.vllm.rotary_embedding_return_tensors(
                    flat_pos, flat_q, flat_k,
                    self.head_dim, cos_sin_cache, True,
                )

                xq = flat_q.view(bs, seqlen, -1, self.head_dim)
                xk = flat_k.view(bs, seqlen, -1, self.head_dim)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )

        # Pass (B, S, H, D) directly to VLLMAttention — no transpose needed.
        # VLLMAttention returns (B, S, H*D) ready for wo.
        output = self.inner_attention(xq, xk, xv)
        return self.wo(output)

    for layer in model.layers.values():
        if hasattr(layer, "attention") and isinstance(layer.attention, GQAttention):
            layer.attention.forward = types.MethodType(
                _patched_forward, layer.attention
            )

    logger.info("Replaced RoPE with vLLM fused CUDA kernel")


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
