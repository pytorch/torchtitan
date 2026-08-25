# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative CP contracts for the K3 attention layers.

Two CP algorithms run at once on disjoint layer kinds: Ulysses on the MLA
layers, KCP on the KDA layers. Each is stated here as a placement pair on the
CP mesh axis plus the preconditions that pair implies, so ``apply_cp_kimi_k3``
resolves a contract per module instead of branching per algorithm.

Only the CP axis is declared. The CP collectives run on plain local tensors
after the TP-wrapped projections, at the same gap the TP plan already strips
DTensor, so TP's own head sharding is not this contract's to describe -- and
declaring both here would be two mesh axes on tensor dim 2, which SpmdLayout
rejects without an explicit partition_spec.
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.models.common.attention import (
    create_attention_mask,
    get_causal_mask_mod,
)


__all__ = [
    "CPContract",
    "KCP",
    "ULYSSES",
    "contract_for_mode",
    "cp_all_to_all_headseq",
]

CP = MeshAxisName.CP

# Tensor dims of the [T, H, K] activations the contracts talk about. This model
# carries a folded token stream with no batch axis, so the sequence is dim 0.
SEQ_DIM = 0
HEAD_DIM = 1


def _cp(axis_type: spmd.PerMeshAxisSpmdType) -> SpmdLayout:
    return SpmdLayout(axis_types={CP: axis_type})


@dataclass(frozen=True, slots=True)
class CPContract:
    """What one CP algorithm does to the [B, T, H, K] activations.

    Attributes:
        name: the mode spelling, and what the wiring log reports.
        in_src: Placement entering the attention body.
        in_dst: Placement the body computes at.
        out_src: Placement leaving the body.
        out_dst: Placement at the module boundary.
        head_sharded: Whether the body splits heads across CP, i.e. whether
            the head-divisibility precondition applies.
    """

    name: str
    in_src: SpmdLayout
    in_dst: SpmdLayout
    out_src: SpmdLayout
    out_dst: SpmdLayout
    head_sharded: bool

    def redistributes(self) -> bool:
        """False when in_dst == in_src, i.e. the boundary moves no data."""
        return self.in_src.axis_types != self.in_dst.axis_types

    def in_dims(self) -> tuple[int, int]:
        """(src, dst) tensor dims the CP axis shards on the way in."""
        return _shard_dim(self.in_src), _shard_dim(self.in_dst)

    def out_dims(self) -> tuple[int, int]:
        """(src, dst) tensor dims the CP axis shards on the way out."""
        return _shard_dim(self.out_src), _shard_dim(self.out_dst)


def _shard_dim(layout: SpmdLayout) -> int:
    axis_type = layout.axis_types[CP]
    if not isinstance(axis_type, spmd.Shard):
        raise ValueError(
            f"CP contract expects a Shard on the CP axis, got {axis_type!r}"
        )
    return axis_type.dim


# Ulysses: projections run seq-local, then one all-to-all trades the sharded
# axis -- sequence for heads -- so the body sees the full sequence for its head
# subset. The output pair is the same swap reversed.
ULYSSES = CPContract(
    name="ulysses",
    in_src=_cp(spmd.S(SEQ_DIM)),
    in_dst=_cp(spmd.S(HEAD_DIM)),
    out_src=_cp(spmd.S(HEAD_DIM)),
    out_dst=_cp(spmd.S(SEQ_DIM)),
    head_sharded=True,
)

# KCP: the sequence stays sharded end to end (report sec 5.1.2). The delta-rule
# recurrence carries state rank to rank -- a sequential dependency no placement
# pair describes -- so the contract is an identity, declared to keep one shape.
KCP = CPContract(
    name="kcp",
    in_src=_cp(spmd.S(SEQ_DIM)),
    in_dst=_cp(spmd.S(SEQ_DIM)),
    out_src=_cp(spmd.S(SEQ_DIM)),
    out_dst=_cp(spmd.S(SEQ_DIM)),
    head_sharded=False,
)

_BY_MODE = {c.name: c for c in (ULYSSES, KCP)}


def contract_for_mode(mode: str) -> CPContract:
    if mode not in _BY_MODE:
        raise ValueError(f"cp mode must be one of {sorted(_BY_MODE)}, got {mode!r}")
    return _BY_MODE[mode]


def cp_all_to_all_headseq(
    x: torch.Tensor, cp_group, *, src_dim: int, dst_dim: int
) -> torch.Tensor:
    """Differentiable Ulysses all-to-all moving the CP shard between tensor dims.

    ``(0, 1)``: ``[T/cp, H, K]`` (seq-sharded) -> ``[T, H/cp, K]``.
    ``(1, 0)``: ``[T, H/cp, K]`` -> ``[T/cp, H, K]``.

    The dims come from the CP contract's placement pair rather than a flag, so a
    contract that names a pair with no implementation raises here instead of being
    quietly ignored.

    Numerics (round-trip and per-head chunk_kda parity) validated
    bit-exact against a single-rank reference; backward is the
    transposed all-to-all via torch.distributed.nn.functional.
    """

    if (src_dim, dst_dim) not in ((SEQ_DIM, HEAD_DIM), (HEAD_DIM, SEQ_DIM)):
        raise ValueError(
            f"no Ulysses all-to-all for CP shard dims {src_dim} -> {dst_dim}; "
            f"implemented pairs are {SEQ_DIM} <-> {HEAD_DIM}"
        )
    cp = dist.get_world_size(cp_group)
    d0, d1, K = x.shape
    if (src_dim, dst_dim) == (SEQ_DIM, HEAD_DIM):
        t_loc, num_heads = d0, d1
        # [T/cp, H, K] -> [cp, T/cp, H/cp, K] (split heads by destination rank)
        x_split = x.reshape(t_loc, cp, num_heads // cp, K).permute(1, 0, 2, 3)
        out = dist_nn.all_to_all_single(
            torch.empty_like(x_split.contiguous()), x_split.contiguous(), group=cp_group
        )
        # recv[s] holds source s's T/cp for THIS rank's head subset, and s is
        # already the sequence-chunk order, so the reshape stacks the sequence.
        return out.reshape(cp * t_loc, num_heads // cp, K).contiguous()
    t_full, h_loc = d0, d1
    t_loc = t_full // cp
    # dim 0 is the destination rank: which sequence chunk each rank receives.
    x_split = x.reshape(cp, t_loc, h_loc, K).contiguous()
    out = dist_nn.all_to_all_single(torch.empty_like(x_split), x_split, group=cp_group)
    # out[s] = source s's head subset for THIS rank's sequence chunk; put T/cp
    # first so the reshape stacks heads in ascending source order.
    return out.permute(1, 0, 2, 3).reshape(t_loc, cp * h_loc, K).contiguous()


def full_sequence_causal_mask(attn, num_tokens: int, device):
    """Causal-only mask for the sequence Ulysses reassembles, cached on
    ``attn`` per (length, device). Correct only while the folded stream holds
    ONE document, so a stream wider than the context window is rejected -- a
    causal-only rebuild cannot see document boundaries."""
    limit = getattr(attn, "_cp_max_context_length", None)
    if limit is not None and num_tokens > limit:
        raise NotImplementedError(
            f"context parallel folds {num_tokens} tokens into one stream "
            f"but the context window is {limit}, so the "
            "stream holds more than one document. The CP path rebuilds a "
            "causal-only mask and cannot see document boundaries; use a "
            "microbatch no wider than the context window."
        )
    key = (num_tokens, device)
    if attn._cp_mask is None or attn._cp_mask[0] != key:
        mask = create_attention_mask(
            get_causal_mask_mod(), None, None, num_tokens, num_tokens, device=device
        )
        attn._cp_mask = (key, mask)
    return attn._cp_mask[1]


def mla_ulysses_attention(
    attn,
    q_LHQ: torch.Tensor,
    kv_LHC: torch.Tensor,
    k_rope_LR: torch.Tensor,
    cp_group,
) -> torch.Tensor:
    """MLA attention over the full sequence for this rank's head subset.

    * One fused all-to-all trades the sharded axis, sequence for heads; the
      attention backend runs unchanged; a second all-to-all trades back.
    * The rotary slice stays OUT of the exchange: it is headless (one vector
      per token), so it is all-gathered along the sequence and expanded onto
      local heads. Packing the expanded key instead reassembles it against the
      wrong head subset.
    * Shape suffixes beyond the legend: L local sequence (T/cp), G this rank's
      head count, W packed channel width, R rotary width.
    """
    cp_size = dist.get_world_size(cp_group)
    t_loc = q_LHQ.shape[0]
    t_full = t_loc * cp_size
    # q_LHQ already carries this rank's TP-local heads, so cp splits those.
    h_cp = q_LHQ.shape[1] // cp_size

    packed_LHW = torch.cat([q_LHQ, kv_LHC], dim=-1)
    src_dim, dst_dim = ULYSSES.in_dims()
    packed_TGW = cp_all_to_all_headseq(
        packed_LHW, cp_group, src_dim=src_dim, dst_dim=dst_dim
    )
    q_TGQ, k_nope_TGN, v_TGV = torch.split(
        packed_TGW,
        [attn.q_head_dim, attn.qk_nope_head_dim, attn.v_head_dim],
        dim=-1,
    )

    # Differentiable all-gather: backward is the reduce-scatter a value every
    # rank consumed needs.
    k_rope_TR = torch.cat(
        dist_nn.all_gather(k_rope_LR.contiguous(), group=cp_group), dim=0
    )
    k_TGQ = torch.cat(
        [
            k_nope_TGN,
            k_rope_TR.view(t_full, 1, attn.qk_rope_head_dim).expand(
                t_full, h_cp, attn.qk_rope_head_dim
            ),
        ],
        dim=-1,
    )

    out_TGV = attn.inner_attention(
        q_TGQ,
        k_TGQ,
        v_TGV,
        attention_masks=full_sequence_causal_mask(attn, t_full, q_TGQ.device),
        scale=attn.scale,
    )
    out_src_dim, out_dst_dim = ULYSSES.out_dims()
    return cp_all_to_all_headseq(
        out_TGV.contiguous(), cp_group, src_dim=out_src_dim, dst_dim=out_dst_dim
    )
