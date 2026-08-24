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

See CP_DECLARATIVE.md in the logbook for why KCP is an identity pair.
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.distributed as dist

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout


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
        name: ``kda_cp_mode`` spelling, and what the wiring log reports.
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
# recurrence carries state rank to rank, which is a sequential dependency, not a
# redistribution -- no placement pair describes it, so it stays inside the op and
# the contract is an identity. Declared anyway to keep one shape for both modes.
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
        raise ValueError(f"kda_cp_mode must be one of {sorted(_BY_MODE)}, got {mode!r}")
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
    import torch.distributed.nn.functional as dist_nn

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
