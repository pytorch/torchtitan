# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native-style 2D model paths for the inference-gap ablation.

The profiler proof (docs/inference_gap_ablation.md) showed the residual AR gap is
cross-rank ARRIVAL SPIN, not a slower kernel: pure comm is identical to native,
but torchtitan's ranks arrive at each collective spread out (5.5x more spin)
because of the per-rank-variable work the DTensor 3D path inserts between
collectives. Native closes the gap because it runs a uniform, minimal 2D path on
plain local tensors with one AR backend.

This module ports the native-style 2D path two ways, swapped in over the built
Qwen3 model after weight load (see apply_2d_model):

* "local"  -- PURE-LOCAL: the whole forward runs on plain local tensors (drop to
  local right after the embedding, never rewrap to DTensor; local F.linear +
  vllm all-reduce; pre-extracted local weight shards). Mirrors native's
  structure exactly. No DTensor dispatch in the forward.
* "dtensor" -- the original qwen3_vllm.py design (commit ca732d7e9): stays in
  DTensor through norms/rope/attention/silu via register_sharding, dropping to
  local only at QKV and wo/w2. Carries the register_sharding cost (measured
  -26% in isolation on this build) -- kept for the A/B comparison.

Both are dense-only (Qwen3-32B / 1.7B); MoE blocks are not supported here.
"""
from __future__ import annotations

import logging

import spmd_types as spmd

import torch
import torch.distributed as _torch_dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding

from torchtitan.models.qwen3.model import Qwen3TransformerBlock

from vllm.distributed import tensor_model_parallel_all_reduce

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local(t: torch.Tensor) -> torch.Tensor:
    """Local shard of a DTensor weight (identity for plain tensors).

    Detached so the stored fused/local weight is a leaf (requires_grad=False):
    inference-only, and nn.Module attribute assignment rejects non-leaf tensors
    (to_local() carries a grad_fn).
    """
    local = t.to_local() if isinstance(t, DTensor) else t
    return local.detach()


def _convert_rope_cache_to_vllm(
    rope_cache: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """TorchTitan CosSinRoPE cache (max_seq, 2*head_dim), layout
    [cos(cat(f,f)), sin(cat(f,f))], to vLLM's (max_seq, head_dim) [cos(f),sin(f)].
    """
    if isinstance(rope_cache, DTensor):
        rope_cache = rope_cache.to_local()
    rope_cache = rope_cache.detach()
    head_dim = rope_cache.shape[-1] // 2
    half = head_dim // 2
    cos_freqs = rope_cache[:, :half]
    sin_freqs = rope_cache[:, head_dim : head_dim + half]
    result = torch.cat([cos_freqs, sin_freqs], dim=-1)
    return result.to(dtype) if dtype is not None else result


# ---------------------------------------------------------------------------
# Custom ops (opaque to torch.compile / cudagraph), bodies from ca732d7e9
# ---------------------------------------------------------------------------

_VLLM_ATTN_MODULES: dict[str, torch.nn.Module] = {}


@torch.library.custom_op("tt2d::vllm_attention", mutates_args=())
def vllm_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layer_name: str
) -> torch.Tensor:
    """Opaque wrapper around vLLM's paged Attention. q/k/v: [T, H, D].
    Returns [T, H*D] (flash-attn token padding trimmed)."""
    attn = _VLLM_ATTN_MODULES[layer_name]
    out = attn(query, key, value)
    return out.narrow(0, 0, query.shape[0])


@vllm_attention.register_fake
def _vllm_attention_fake(query, key, value, layer_name):
    return torch.empty(
        query.shape[0],
        query.shape[1] * query.shape[2],
        dtype=query.dtype,
        device=query.device,
    )


@torch.library.custom_op("tt2d::rotary_embedding", mutates_args=())
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque wrapper around vLLM's in-place _C::rotary_embedding (cloned so the
    mutation is not functionalized). q/k: [T, H*D]."""
    q_out = query.clone()
    k_out = key.clone()
    torch.ops._C.rotary_embedding(
        positions, q_out, k_out, head_size, cos_sin_cache, True
    )
    return q_out, k_out


@rotary_embedding.register_fake
def _rotary_embedding_fake(positions, query, key, head_size, cos_sin_cache):
    return query.clone(), key.clone()


@torch.library.custom_op("tt2d::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Opaque wrapper around vLLM's _C::silu_and_mul. [T, 2D] -> [T, D]."""
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


@silu_and_mul.register_fake
def _silu_and_mul_fake(x):
    d = x.shape[-1] // 2
    return torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)


@torch.library.custom_op("tt2d::add_rms_norm_", mutates_args=("hidden", "residual"))
def add_rms_norm_(
    hidden: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """In-place fused residual-add + RMSNorm (native vLLM pattern, no clones).

    After the call: ``residual = residual + hidden`` (the running residual) and
    ``hidden = rms_norm(residual) * weight`` (the normed activation).
    """
    torch.ops._C.fused_add_rms_norm(hidden, residual, weight, eps)


@add_rms_norm_.register_fake
def _add_rms_norm_fake(hidden, residual, weight, eps) -> None:
    return None


def _add_rmsnorm(hidden, residual, weight, eps):
    """Functional wrapper: returns (normed, new_residual) after the in-place op."""
    add_rms_norm_(hidden, residual, weight, eps)
    return hidden, residual


# -- DTensor sharding strategies (only used by the "dtensor" variant) --------


def _register_2d_shardings() -> None:
    @register_sharding(torch.ops.tt2d.rotary_embedding.default)
    def _rope_strategy(positions, query, key, head_size, cos_sin_cache):
        # positions/cache Replicate; q/k [T, H*D] Shard(1) on heads.
        return [
            (
                [Replicate(), Replicate()],
                [Replicate(), Replicate(), Replicate(), None, Replicate()],
            ),
            (
                [Shard(1), Shard(1)],
                [Replicate(), Shard(1), Shard(1), None, Replicate()],
            ),
        ]

    @register_sharding(torch.ops.tt2d.vllm_attention.default)
    def _attn_strategy(query, key, value, layer_name):
        # q/k/v [T, H, D] Shard(1) on heads; out [T, H*D] Shard(1).
        return [
            ([Replicate()], [Replicate(), Replicate(), Replicate(), None]),
            ([Shard(1)], [Shard(1), Shard(1), Shard(1), None]),
        ]

    @register_sharding(torch.ops.tt2d.silu_and_mul.default)
    def _silu_strategy(x):
        return [([Replicate()], [Replicate()]), ([Shard(1)], [Shard(1)])]


# ---------------------------------------------------------------------------
# Block variants (attributes set by prepare_2d; dense FFN only)
# ---------------------------------------------------------------------------


class Qwen3VLLMBlockLocal(Qwen3TransformerBlock):
    """Pure-local 2D block: x is a plain local [T, D] tensor throughout."""

    def forward(self, x, positions):
        x = x + self._attn(x, positions)
        x = x + self._ffn(x)
        return x

    def _attn(self, x, positions):
        normed = F.rms_norm(x, (x.shape[-1],), self._attn_norm_w, self._attn_norm_eps)
        return self._attn_core(normed, positions)

    def _ffn(self, x):
        normed = F.rms_norm(x, (x.shape[-1],), self._ffn_norm_w, self._ffn_norm_eps)
        return self._ffn_core(normed)

    # Post-norm cores (take already-normed input) -- shared with the fused variant.
    def _attn_core(self, normed, positions):
        attn = self.attention
        t = normed.shape[0]
        xqkv = F.linear(normed, attn._fused_qkv_weight).view(t, -1, attn.head_dim)
        xq, xk, xv = xqkv.split(
            [attn._nq_local, attn._nkv_local, attn._nkv_local], dim=1
        )
        if attn.q_norm is not None:
            xq = F.rms_norm(xq, (attn.head_dim,), attn._q_norm_w, attn._q_norm_eps)
            xk = F.rms_norm(xk, (attn.head_dim,), attn._k_norm_w, attn._k_norm_eps)
        xq, xk = rotary_embedding(
            positions,
            xq.reshape(t, -1),
            xk.reshape(t, -1),
            attn.head_dim,
            attn._vllm_cos_sin_cache,
        )
        out = vllm_attention(
            xq.view(t, -1, attn.head_dim),
            xk.view(t, -1, attn.head_dim),
            xv,
            attn._vllm_layer_name,
        )
        h = F.linear(out, attn._local_wo_weight)
        return self._all_reduce(h)

    def _ffn_core(self, normed):
        ff = self.feed_forward
        gate_up = F.linear(normed, ff._fused_gate_up_weight_local)
        h = silu_and_mul(gate_up)
        h = F.linear(h, ff._local_w2_weight)
        return self._all_reduce(h)

    def _all_reduce(self, h: torch.Tensor) -> torch.Tensor:
        """Partial -> Replicate over the TP group. Pure-local uses vLLM's custom
        all-reduce; the spmd_types subclass overrides this to use spmd's."""
        return tensor_model_parallel_all_reduce(h)


class Qwen3VLLMBlockLocalFused(Qwen3VLLMBlockLocal):
    """Pure-local with native-style residual fusion: the residual is threaded
    across layers and every residual-add is fused into the next RMSNorm via
    vLLM's in-place fused_add_rms_norm (no standalone add / memcpy kernels).
    """

    def forward(self, hidden, residual, positions):
        if residual is None:
            # First layer: no preceding residual to add; plain RMSNorm.
            residual = hidden
            normed = F.rms_norm(
                hidden, (hidden.shape[-1],), self._attn_norm_w, self._attn_norm_eps
            )
        else:
            normed, residual = _add_rmsnorm(
                hidden, residual, self._attn_norm_w, self._attn_norm_eps
            )
        attn_out = self._attn_core(normed, positions)
        normed, residual = _add_rmsnorm(
            attn_out, residual, self._ffn_norm_w, self._ffn_norm_eps
        )
        ffn_out = self._ffn_core(normed)
        return ffn_out, residual


# Process group for spmd_types redistribute, set in _prepare_2d.
_SPMD_REDIST_GROUP = None


class _VLLMAllReduceDist:
    """A ``torch.distributed``-like shim that routes ``all_reduce`` through
    vLLM's custom (one-shot/multimem) all-reduce instead of NCCL, so
    ``spmd_types.redistribute(P -> R)`` uses the same fast collective as the
    pure-local path. Every other attribute delegates to ``torch.distributed``.

    Installed process-wide via ``spmd.set_dist`` for the ``spmd`` variant.
    spmd's all_reduce is in-place (``all_reduce(result, ...)``), so we write
    vLLM's out-of-place result back into the buffer.
    """

    ReduceOp = _torch_dist.ReduceOp

    @staticmethod
    def all_reduce(tensor, op=_torch_dist.ReduceOp.SUM, group=None, async_op=False):
        tensor.copy_(tensor_model_parallel_all_reduce(tensor))
        return None

    def __getattr__(self, name):
        return getattr(_torch_dist, name)


class Qwen3VLLMBlockSpmdFused(Qwen3VLLMBlockLocalFused):
    """spmd_types execution: identical plain-local fused forward as the pure-local
    fused block, but the Partial -> Replicate all-reduce after wo/w2 goes through
    spmd_types' redistribute (typecheck off) instead of vLLM's custom kernel.
    """

    def _all_reduce(self, h: torch.Tensor) -> torch.Tensor:
        return spmd.redistribute(h, _SPMD_REDIST_GROUP, src=spmd.P, dst=spmd.R)


class Qwen3VLLMBlockLocal3D(Qwen3TransformerBlock):
    """Pure-local but 3D: residual/linears/norms stay (1, T, D); flatten to
    token-major only at the rope/attention kernels (which require it).

    Isolates the tensor-rank variable vs Qwen3VLLMBlockLocal: same no-DTensor
    pure-local collectives, but the big GEMMs run as 3D aten::linear (view+mm+view
    under torch.compile) instead of a direct 2D mm.
    """

    def forward(self, x, positions):
        x = x + self._attn(x, positions)
        x = x + self._ffn(x)
        return x

    def _attn(self, x, positions):
        attn = self.attention
        bs, t, _ = x.shape  # bs == 1
        normed = F.rms_norm(x, (x.shape[-1],), self._attn_norm_w, self._attn_norm_eps)
        xqkv = F.linear(normed, attn._fused_qkv_weight)  # (1, T, qkv) 3D linear
        xqkv = xqkv.view(bs * t, -1, attn.head_dim)  # flatten tokens for kernels
        xq, xk, xv = xqkv.split(
            [attn._nq_local, attn._nkv_local, attn._nkv_local], dim=1
        )
        if attn.q_norm is not None:
            xq = F.rms_norm(xq, (attn.head_dim,), attn._q_norm_w, attn._q_norm_eps)
            xk = F.rms_norm(xk, (attn.head_dim,), attn._k_norm_w, attn._k_norm_eps)
        xq, xk = rotary_embedding(
            positions,
            xq.reshape(bs * t, -1),
            xk.reshape(bs * t, -1),
            attn.head_dim,
            attn._vllm_cos_sin_cache,
        )
        out = vllm_attention(
            xq.view(bs * t, -1, attn.head_dim),
            xk.view(bs * t, -1, attn.head_dim),
            xv,
            attn._vllm_layer_name,
        )
        h = F.linear(out.view(bs, t, -1), attn._local_wo_weight)  # (1, T, D) 3D
        return tensor_model_parallel_all_reduce(h)

    def _ffn(self, x):
        ff = self.feed_forward
        normed = F.rms_norm(x, (x.shape[-1],), self._ffn_norm_w, self._ffn_norm_eps)
        gate_up = F.linear(normed, ff._fused_gate_up_weight_local)  # (1, T, 2F) 3D
        h = silu_and_mul(gate_up)  # (1, T, F)
        h = F.linear(h, ff._local_w2_weight)  # (1, T, D) 3D
        return tensor_model_parallel_all_reduce(h)


class Qwen3VLLMBlockDTensor(Qwen3TransformerBlock):
    """2D-DTensor block (qwen3_vllm.py design): DTensor [T, D] residual, local
    only at fused-QKV and wo/w2 + all-reduce."""

    def forward(self, x, positions):
        x = x + self._attn(self.attention_norm(x), positions)
        x = x + self._ffn(self.ffn_norm(x))
        return x

    def _attn(self, x, positions):
        attn = self.attention
        mesh = x.device_mesh
        t = x.shape[0]
        xqkv = F.linear(x.to_local(), attn._fused_qkv_weight).view(t, -1, attn.head_dim)
        xq, xk, xv = xqkv.split(
            [attn._nq_local, attn._nkv_local, attn._nkv_local], dim=1
        )
        xq = DTensor.from_local(xq, mesh, [Shard(1)])
        xk = DTensor.from_local(xk, mesh, [Shard(1)])
        xv = DTensor.from_local(xv, mesh, [Shard(1)])
        if attn.q_norm is not None:
            xq = F.rms_norm(
                xq, attn.q_norm.normalized_shape, attn.q_norm.weight, attn.q_norm.eps
            )
            xk = F.rms_norm(
                xk, attn.k_norm.normalized_shape, attn.k_norm.weight, attn.k_norm.eps
            )
        if positions is not None and not isinstance(positions, DTensor):
            positions = DTensor.from_local(positions, mesh, [Replicate()])
        xq, xk = rotary_embedding(
            positions,
            xq.reshape(t, -1),
            xk.reshape(t, -1),
            attn.head_dim,
            attn._vllm_cos_sin_cache_dt,
        )
        out = vllm_attention(
            xq.view(t, -1, attn.head_dim),
            xk.view(t, -1, attn.head_dim),
            xv,
            attn._vllm_layer_name,
        )
        h = F.linear(out.to_local(), attn._local_wo_weight)
        h = tensor_model_parallel_all_reduce(h)
        return DTensor.from_local(h, mesh, [Replicate()])

    def _ffn(self, x):
        ff = self.feed_forward
        mesh = x.device_mesh
        gate_up = F.linear(x, ff._fused_gate_up_weight_dt)
        h = silu_and_mul(gate_up)
        h = F.linear(h.to_local(), ff._local_w2_weight)
        h = tensor_model_parallel_all_reduce(h)
        return DTensor.from_local(h, mesh, [Replicate()])


# ---------------------------------------------------------------------------
# Prepare + forward + apply
# ---------------------------------------------------------------------------


def _prepare_2d(wrapper, variant: str) -> None:
    """Extract fused local weights, vLLM attn modules, cos/sin cache; swap block
    classes. Runs after weight load, before any (compiled) forward."""
    model = wrapper.model
    tp_enabled = wrapper.parallel_dims.tp_enabled
    tp_group_name = None
    tp_rank = 0
    if tp_enabled:
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        tp_group_name = tp_group.unique_name
        tp_rank = tp_group.rank_in_group

    first_attn = next(iter(model.layers.values())).attention
    if not hasattr(first_attn.qkv_linear, "wq"):
        raise ValueError(
            "2D model path requires QKVLinear (separate wq/wk/wv); "
            "FusedQKVLinear is not supported."
        )
    mesh = first_attn.qkv_linear.wq.weight.device_mesh
    cos_sin_local = _convert_rope_cache_to_vllm(
        first_attn.rope.cache, dtype=torch.bfloat16
    )
    cos_sin_dt = DTensor.from_local(cos_sin_local, mesh, [Replicate()])

    block_cls = {
        "local": Qwen3VLLMBlockLocal,
        "localfused": Qwen3VLLMBlockLocalFused,
        "spmd": Qwen3VLLMBlockSpmdFused,
        "local3d": Qwen3VLLMBlockLocal3D,
        "dtensor": Qwen3VLLMBlockDTensor,
    }[variant]

    if variant == "spmd":
        # spmd_types redistribute targets this TP process group (the axis the
        # weights are sharded on). typecheck stays OFF for the perf run.
        global _SPMD_REDIST_GROUP
        _SPMD_REDIST_GROUP = mesh.get_group()
        # Route spmd's all_reduce through vLLM's custom AR (not NCCL) so the
        # redistribute matches the pure-local collective.
        spmd.set_dist(_VLLMAllReduceDist())

    for name, layer in model.layers.items():
        if getattr(layer, "moe_enabled", False):
            raise ValueError("2D model path supports dense FFN blocks only (no MoE).")
        attn = layer.attention
        ff = layer.feed_forward
        qkv = attn.qkv_linear

        wq, wk, wv = _local(qkv.wq.weight), _local(qkv.wk.weight), _local(qkv.wv.weight)
        attn._fused_qkv_weight = torch.cat([wq, wk, wv], dim=0)
        attn._nq_local = wq.shape[0] // attn.head_dim
        attn._nkv_local = wk.shape[0] // attn.head_dim
        attn._local_wo_weight = _local(attn.wo.weight)
        if attn.q_norm is not None:
            attn._q_norm_w = _local(attn.q_norm.weight)
            attn._k_norm_w = _local(attn.k_norm.weight)
            attn._q_norm_eps = attn.q_norm.eps
            attn._k_norm_eps = attn.k_norm.eps
        attn._vllm_cos_sin_cache = cos_sin_local
        attn._vllm_cos_sin_cache_dt = cos_sin_dt

        vllm_layer_name = f"model.layers.{name}.attention.inner_attention"
        attn._vllm_layer_name = vllm_layer_name
        _VLLM_ATTN_MODULES[vllm_layer_name] = attn.inner_attention.vllm_attn

        gate_up_local = torch.cat([_local(ff.w1.weight), _local(ff.w3.weight)], dim=0)
        ff._fused_gate_up_weight_local = gate_up_local
        ff._fused_gate_up_weight_dt = DTensor.from_local(
            gate_up_local, mesh, [Shard(0)]
        )
        ff._local_w2_weight = _local(ff.w2.weight)

        layer._attn_norm_w = _local(layer.attention_norm.weight)
        layer._ffn_norm_w = _local(layer.ffn_norm.weight)
        layer._attn_norm_eps = layer.attention_norm.eps
        layer._ffn_norm_eps = layer.ffn_norm.eps
        layer.__class__ = block_cls

    # Final norm + embedding locals (pure-local path).
    wrapper._norm_w = _local(model.norm.weight)
    wrapper._norm_eps = model.norm.eps
    emb = model.tok_embeddings
    emb._vllm_local_weight = _local(emb.weight)
    if tp_enabled:
        chunk = (
            emb.num_embeddings + wrapper.parallel_dims.tp - 1
        ) // wrapper.parallel_dims.tp
        emb._vllm_offset = tp_rank * chunk
    else:
        emb._vllm_offset = 0

    logger.info(
        "Prepared 2D %s model (%d layers, tp_group=%s)",
        variant,
        len(model.layers),
        tp_group_name,
    )


def _forward_local(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
    """Pure-local 2D forward: plain local tensors end-to-end."""
    if inputs_embeds is not None:
        raise NotImplementedError("inputs_embeds not yet supported")
    if input_ids is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")

    emb = self.model.tok_embeddings
    w = emb._vllm_local_weight
    offset = emb._vllm_offset
    mask = (input_ids >= offset) & (input_ids < offset + w.shape[0])
    local_ids = (input_ids - offset).clamp(0, w.shape[0] - 1)
    h = F.embedding(local_ids, w) * mask.unsqueeze(-1).to(w.dtype)
    h = tensor_model_parallel_all_reduce(h)  # [T, D] local, replicated

    for layer in self.model.layers.values():
        h = layer(h, positions)

    h = F.rms_norm(h, (h.shape[-1],), self._norm_w, self._norm_eps)
    return h


def _forward_localfused(
    self, input_ids=None, positions=None, inputs_embeds=None, **kwargs
):
    """Pure-local forward with native-style residual fusion (threaded residual,
    fused add+RMSNorm)."""
    if inputs_embeds is not None:
        raise NotImplementedError("inputs_embeds not yet supported")
    if input_ids is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")

    emb = self.model.tok_embeddings
    w = emb._vllm_local_weight
    offset = emb._vllm_offset
    mask = (input_ids >= offset) & (input_ids < offset + w.shape[0])
    local_ids = (input_ids - offset).clamp(0, w.shape[0] - 1)
    hidden = F.embedding(local_ids, w) * mask.unsqueeze(-1).to(w.dtype)
    hidden = tensor_model_parallel_all_reduce(hidden)  # [T, D] local, replicated

    residual = None
    for layer in self.model.layers.values():
        hidden, residual = layer(hidden, residual, positions)

    # Final norm fuses the last residual add (matches native's model.norm).
    normed, _ = _add_rmsnorm(hidden, residual, self._norm_w, self._norm_eps)
    return normed


def _forward_spmd(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
    """spmd_types forward: same as _forward_localfused but the vocab-parallel
    embedding all-reduce uses spmd_types' redistribute (NCCL) instead of vLLM's
    custom all-reduce. The block all-reduces use Qwen3VLLMBlockSpmdFused."""
    if inputs_embeds is not None:
        raise NotImplementedError("inputs_embeds not yet supported")
    if input_ids is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")

    emb = self.model.tok_embeddings
    w = emb._vllm_local_weight
    offset = emb._vllm_offset
    mask = (input_ids >= offset) & (input_ids < offset + w.shape[0])
    local_ids = (input_ids - offset).clamp(0, w.shape[0] - 1)
    hidden = F.embedding(local_ids, w) * mask.unsqueeze(-1).to(w.dtype)
    hidden = spmd.redistribute(hidden, _SPMD_REDIST_GROUP, src=spmd.P, dst=spmd.R)

    residual = None
    for layer in self.model.layers.values():
        hidden, residual = layer(hidden, residual, positions)

    normed, _ = _add_rmsnorm(hidden, residual, self._norm_w, self._norm_eps)
    return normed


def _forward_local3d(
    self, input_ids=None, positions=None, inputs_embeds=None, **kwargs
):
    """Pure-local forward keeping the (1, T, D) shape (3D linears)."""
    if inputs_embeds is not None:
        raise NotImplementedError("inputs_embeds not yet supported")
    if input_ids is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")

    emb = self.model.tok_embeddings
    w = emb._vllm_local_weight
    offset = emb._vllm_offset
    mask = (input_ids >= offset) & (input_ids < offset + w.shape[0])
    local_ids = (input_ids - offset).clamp(0, w.shape[0] - 1)
    h = F.embedding(local_ids, w) * mask.unsqueeze(-1).to(w.dtype)
    h = tensor_model_parallel_all_reduce(h).unsqueeze(0)  # (1, T, D) local

    for layer in self.model.layers.values():
        h = layer(h, positions)

    h = F.rms_norm(h, (h.shape[-1],), self._norm_w, self._norm_eps)
    return h.view(-1, h.shape[-1])


def _forward_dtensor(
    self, input_ids=None, positions=None, inputs_embeds=None, **kwargs
):
    """2D-DTensor forward (qwen3_vllm.py design)."""
    if inputs_embeds is not None:
        raise NotImplementedError("inputs_embeds not yet supported")
    if input_ids is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")

    h = self.model.tok_embeddings(input_ids)  # DTensor [T, D]
    if isinstance(h, DTensor) and any(not p.is_replicate() for p in h.placements):
        h = h.redistribute(placements=[Replicate()] * h.device_mesh.ndim)

    for layer in self.model.layers.values():
        h = layer(h, positions)

    h = self.model.norm(h)
    if isinstance(h, DTensor):
        h = h.full_tensor()
    if h.dim() == 3:
        h = h.view(-1, h.size(-1))
    return h


def apply_2d_model(variant: str) -> None:
    """Patch VLLMModelWrapper to swap in the 2D model after weight load and use
    the 2D forward. variant: "local" (pure-local) or "dtensor" (register_sharding).
    """
    if variant not in ("local", "localfused", "spmd", "local3d", "dtensor"):
        raise ValueError(f"Unknown 2D model variant {variant!r}")

    from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper

    if variant == "dtensor":
        _register_2d_shardings()

    _orig_init = VLLMModelWrapper.__init__

    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        _prepare_2d(self, variant)

    VLLMModelWrapper.__init__ = patched_init
    VLLMModelWrapper.forward = {
        "local": _forward_local,
        "localfused": _forward_localfused,
        "spmd": _forward_spmd,
        "local3d": _forward_local3d,
        "dtensor": _forward_dtensor,
    }[variant]
    logger.info("apply_2d_model: patched VLLMModelWrapper for variant=%s", variant)
