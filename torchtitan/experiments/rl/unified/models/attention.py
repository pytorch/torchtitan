# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import types

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard
from torchtitan.experiments.rl.vllm_compat.models.attention import (
    VLLMCompatibleFlashAttention,
)
from vllm.model_executor.layers.attention.attention import Attention

logger = logging.getLogger(__name__)


def _local(t: torch.Tensor) -> torch.Tensor:
    """Return the local tensor if DTensor, otherwise identity."""
    return t._local_tensor if isinstance(t, DTensor) else t


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
        # Capture the original symbolic seq_len from the input BEFORE
        # to_local() so that the symbol is the same one GQAttention uses
        # in its .view(bs, seqlen, -1) call.
        batch_size, _, seq_len, head_dim = q.shape

        # Unwrap DTensor inputs to local tensors for attention computation.
        # After vLLM attention, we always re-wrap with Shard(1) because
        # ColwiseParallel on wq/wk/wv shards heads across TP ranks, so the
        # output is always head-sharded on dim=1 of (batch, heads, seq, head_dim).
        device_mesh = None
        if isinstance(q, DTensor):
            device_mesh = q.device_mesh
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
                output, device_mesh=device_mesh, placements=(Shard(1),)
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


# ---------------------------------------------------------------------------
# Fused RoPE: replace TorchTitan cos_sin RoPE with vLLM _C::rotary_embedding
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module-level registry for vllm attention modules (keyed by layer name).
# Custom ops can only take tensors/primitives, so we look up the module here.
# ---------------------------------------------------------------------------
_VLLM_ATTN_MODULES: dict[str, torch.nn.Module] = {}


@torch.library.custom_op("torchtitan::vllm_attention", mutates_args=())
def _vllm_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Non-mutating wrapper around vLLM's unified_attention_with_output.

    Treats the entire vllm Attention call as opaque so torch.compile
    doesn't functionalize the internal output-buffer mutation.
    """
    attn = _VLLM_ATTN_MODULES[layer_name]
    out = attn(query, key, value)
    return out.narrow(0, 0, query.shape[0])  # trim flash-attn padding


@_vllm_attention.register_fake
def _vllm_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    # output is (T, num_heads * head_dim)
    return torch.empty(
        query.shape[0],
        query.shape[1] * query.shape[2],
        dtype=query.dtype,
        device=query.device,
    )


@torch.library.custom_op("torchtitan::rotary_embedding", mutates_args=())
def _rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-mutating wrapper around vLLM's _C::rotary_embedding.

    Clones q/k so the in-place kernel doesn't trigger auto_functionalized
    under torch.compile.
    """
    q_out = query.clone()
    k_out = key.clone()
    torch.ops._C.rotary_embedding(
        positions, q_out, k_out, head_size, cos_sin_cache, True
    )
    return q_out, k_out


@_rotary_embedding.register_fake
def _rotary_embedding_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query.clone(), key.clone()


@torch.library.custom_op("torchtitan::silu_and_mul", mutates_args=())
def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Non-mutating wrapper around vLLM's _C::silu_and_mul.

    Input: (T, 2*D) — first half is gate, second half is up.
    Output: (T, D) — silu(gate) * up.
    """
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


@_silu_and_mul.register_fake
def _silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)


def _convert_rope_cache_to_vllm(
    rope_cache: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Convert TorchTitan cos_sin rope cache to vLLM format.

    TorchTitan (cos_sin backend): ``(max_seq, 2*head_dim)``
        Layout: ``[cos(cat(f,f)), sin(cat(f,f))]`` — each half is duplicated.

    vLLM: ``(max_seq, head_dim)``
        Layout: ``[cos(f), sin(f)]`` — each half has ``head_dim//2`` values.
    """
    if isinstance(rope_cache, DTensor):
        rope_cache = rope_cache._local_tensor

    head_dim = rope_cache.shape[-1] // 2
    half_head = head_dim // 2

    cos_freqs = rope_cache[:, :half_head]
    sin_freqs = rope_cache[:, head_dim : head_dim + half_head]

    result = torch.cat([cos_freqs, sin_freqs], dim=-1)
    if dtype is not None:
        result = result.to(dtype)
    return result


def _gqa_forward_with_vllm_rope(
    self,
    x: torch.Tensor,
    rope_cache: torch.Tensor,
    attention_masks,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """GQAttention forward with fused RoPE, minimal reshapes.

    Only used when ``self.sequence_first`` is True (vLLM inference path).
    Bypasses VLLMAttention.forward to avoid DTensor wrap/unwrap round-trips.

    The ``rope_cache`` arg is ignored; uses precomputed ``self._vllm_cos_sin_cache``.
    """
    bs, seqlen, _ = x.shape
    num_tokens = bs * seqlen

    # --- Unwrap DTensor once from x, stay local through attention ---
    is_dtensor = isinstance(x, DTensor)
    if is_dtensor:
        device_mesh = x.device_mesh
        x_local = x.to_local()
    else:
        x_local = x

    # Fused QKV projection: one matmul + split instead of 3 separate matmuls
    xqkv = F.linear(x_local, self._fused_qkv_weight)
    xq, xk, xv = xqkv.split([self._q_size, self._kv_size, self._kv_size], dim=-1)

    # Reshape to 4D for QK norms: (B, S, H, D)
    xq = xq.view(bs, seqlen, -1, self.head_dim)
    xk = xk.view(bs, seqlen, -1, self.head_dim)
    xv = xv.view(bs, seqlen, -1, self.head_dim)

    # QK norms (Qwen3) — must be 4D for per-head normalization.
    # Bypass the DTensor module hooks (NoParallel wraps local tensors as
    # Replicate then redistributes to Shard(2), halving the head count).
    # Call F.rms_norm directly with local weights instead.
    if self.q_norm is not None:
        xq = F.rms_norm(
            xq,
            self.q_norm.normalized_shape,
            _local(self.q_norm.weight),
            self.q_norm.eps,
        )
    if self.k_norm is not None:
        xk = F.rms_norm(
            xk,
            self.k_norm.normalized_shape,
            _local(self.k_norm.weight),
            self.k_norm.eps,
        )

    # Flatten to 2D for fused RoPE: (T, H*D)
    xq_2d = xq.view(num_tokens, -1)
    xk_2d = xk.view(num_tokens, -1)
    pos_1d = _local(positions).view(-1) if positions is not None else positions

    xq_2d, xk_2d = torch.ops.torchtitan.rotary_embedding(
        pos_1d,
        xq_2d,
        xk_2d,
        self.head_dim,
        self._vllm_cos_sin_cache,
    )

    # View to 3D for vllm attention: (T, H, D) — no intermediate 4D
    q_3d = xq_2d.view(num_tokens, -1, self.head_dim)
    k_3d = xk_2d.view(num_tokens, -1, self.head_dim)
    v_3d = xv.reshape(num_tokens, -1, self.head_dim)

    # Call vllm attention via custom op (bypass DTensor wrapper + auto_functionalized)
    attn_out = torch.ops.torchtitan.vllm_attention(
        q_3d,
        k_3d,
        v_3d,
        self._vllm_layer_name,
    )

    # wo projection + all-reduce on local tensors (bypass DTensor dispatch)
    h = F.linear(attn_out, self._local_wo_weight)
    if self._tp_group_name is not None:
        h = torch.ops.vllm.all_reduce(h, group_name=self._tp_group_name)
    h = h.view(bs, seqlen, -1)

    # Re-wrap as DTensor so the caller's residual add (x + attn_out) works
    if is_dtensor:
        h = DTensor.from_local(h, device_mesh=device_mesh, placements=[Replicate()])
    return h


def replace_rope_with_vllm_rotary(model) -> None:
    """Replace TorchTitan cos_sin RoPE with vLLM's fused ``_C::rotary_embedding``.

    Precomputes the vLLM-format cos_sin_cache from the model's ``freqs_cis``
    and monkey-patches each ``GQAttention`` forward to call the fused CUDA
    kernel instead of ``apply_rotary_emb_cos_sin``.

    Must be called **after** ``replace_with_vllm_attention`` and after the
    RoPE cache has been extended to max_model_len.

    Note: ``_local_wo_weight`` and ``_tp_group_name`` are set later by
    :func:`prepare_local_weights` after weights are loaded.

    Args:
        model: TorchTitan model with ``.layers`` and ``.freqs_cis``.
    """
    cos_sin_cache = _convert_rope_cache_to_vllm(model.freqs_cis, dtype=torch.bfloat16)

    for layer_name, layer in model.layers.items():
        attn = layer.attention
        attn._vllm_cos_sin_cache = cos_sin_cache
        # Placeholders — set by prepare_local_weights() after weight loading
        attn._tp_group_name = None
        attn._local_wo_weight = None
        attn._fused_qkv_weight = None
        attn._q_size = None
        attn._kv_size = None
        # Store layer name for custom op dispatch and register vllm attention
        attn._vllm_layer_name = f"model.layers.{layer_name}.attention.inner_attention"
        _VLLM_ATTN_MODULES[attn._vllm_layer_name] = attn.inner_attention.vllm_attn
        attn.forward = types.MethodType(_gqa_forward_with_vllm_rope, attn)

    logger.info(
        f"Replaced RoPE with vLLM fused _C::rotary_embedding "
        f"({len(model.layers)} layers, cache shape {cos_sin_cache.shape})"
    )


# ---------------------------------------------------------------------------
# Fused FFN: gate_up matmul + _C::silu_and_mul
# ---------------------------------------------------------------------------


def _ffn_forward_fused(self, x: torch.Tensor) -> torch.Tensor:
    """FeedForward.forward with fused gate_up matmul and silu_and_mul.

    Uses pre-concatenated ``_fused_gate_up_weight`` (w1 || w3) for a single
    matmul, then ``torchtitan::silu_and_mul`` for fused SwiGLU activation.

    Unwraps DTensor to local for the fused matmul, re-wraps for w2.
    """
    is_dtensor = isinstance(x, DTensor)
    if is_dtensor:
        device_mesh = x.device_mesh
        x_local = x.to_local()
    else:
        x_local = x

    gate_up = F.linear(x_local, self._fused_gate_up_weight)  # (*, 2*ffn_dim)
    h = torch.ops.torchtitan.silu_and_mul(gate_up)  # (*, ffn_dim)

    # w2 projection + all-reduce on local tensors (bypass DTensor dispatch)
    h = F.linear(h, self._local_w2_weight)
    if self._tp_group_name is not None:
        h = torch.ops.vllm.all_reduce(h, group_name=self._tp_group_name)

    # Re-wrap as DTensor so the caller's residual add (x + ffn_out) works
    if is_dtensor:
        h = DTensor.from_local(h, device_mesh=device_mesh, placements=[Replicate()])
    return h


def replace_ffn_with_fused(model, tp_group_name: str | None = None) -> None:
    """Fuse gate+up weights and replace FFN forward with fused SwiGLU.

    Pre-concatenates ``w1.weight`` and ``w3.weight`` into a single
    ``(2*ffn_dim, dim)`` matrix, and patches ``FeedForward.forward``
    to use one matmul + ``_C::silu_and_mul`` instead of two matmuls +
    ``silu`` + ``mul``.

    Must be called **after** parallelization and weight loading (so we cat
    the TP-sharded weights with real values).

    Args:
        model: TorchTitan model with ``.layers``.
        tp_group_name: vLLM TP group name for ``vllm::all_reduce`` (None → skip).
    """
    for layer in model.layers.values():
        ff = layer.feed_forward
        w1 = _local(ff.w1.weight)
        w3 = _local(ff.w3.weight)
        ff._fused_gate_up_weight = torch.cat([w1, w3], dim=0)
        ff._local_w2_weight = _local(ff.w2.weight)
        ff._tp_group_name = tp_group_name
        ff.forward = types.MethodType(_ffn_forward_fused, ff)

    logger.info(
        f"Replaced FFN with fused gate_up + silu_and_mul "
        f"({len(model.layers)} layers)"
    )


def prepare_local_weights(model, tp_group_name: str | None = None) -> None:
    """Store local weight references and TP group name for fused forwards.

    Must be called **after** weight loading so the local tensors point to
    real (loaded) values.

    Sets ``_local_wo_weight`` and ``_tp_group_name`` on each attention module.

    Args:
        model: TorchTitan model with ``.layers``.
        tp_group_name: vLLM TP group name for ``vllm::all_reduce`` (None → skip).
    """
    for layer in model.layers.values():
        attn = layer.attention
        attn._local_wo_weight = _local(attn.wo.weight)
        attn._tp_group_name = tp_group_name

        # Fused QKV weight: cat local wq/wk/wv into one matrix
        wq = _local(attn.wq.weight)
        wk = _local(attn.wk.weight)
        wv = _local(attn.wv.weight)
        attn._fused_qkv_weight = torch.cat([wq, wk, wv], dim=0)
        attn._q_size = wq.shape[0]  # local Q output dim (already TP-sharded)
        attn._kv_size = wk.shape[0]  # local KV output dim (already TP-sharded)

    logger.info(
        f"Prepared local weights for fused forwards "
        f"({len(model.layers)} layers, tp_group={tp_group_name})"
    )
