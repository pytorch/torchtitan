# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel kernels for the Kimi K3 attention layers.

Each kernel owns its CP collectives, the shape of
``torchtitan.models.common.cp_attention``: the inner-attention boundary keeps
the cp axis token-sharded on both sides and the kernel's ``forward`` issues the
exchange its algorithm needs.

  - MLA: the generic Ulysses and all-gather kernels specialised to MLA's key.
    MLA expands one rotary vector per token onto every head before the kernel,
    so the expanded key carries ``H`` copies of it; these kernels split the key
    back, move the nope part packed with q and v (Ulysses) or with v (all-gather),
    move the rotary slice once, and expand after the exchange.
  - KDA: Attention Gym's context-parallel delta rule (KCP). The sequence stays
    sharded end to end and the recurrence hands its state from rank to rank.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K = N + R`` the qk head dim (nope
and rope), ``V`` the v head dim, ``W`` a packed width.
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.distributed as dist
from attn_gym.linear.context_parallel import ContextParallelPlan, ContextParallelRouting

from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.cp_attention import (
    AllGatherCPFlexAttention,
    ContextParallelKernel,
    UlyssesCPFlexAttention,
)

from .kda import InnerKDA

__all__ = [
    "ContextParallelInnerKDA",
    "MLAAllGatherCPFlexAttention",
    "MLAUlyssesCPFlexAttention",
    "kcp_routing",
]

_SEQ_DIM = 0
_HEAD_DIM = 1


def _split_rope(
    k_THK: torch.Tensor, rope_head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The nope part ``[T, H, N]`` and the headless rope slice ``[T, R]``."""
    nope_head_dim = k_THK.shape[-1] - rope_head_dim
    k_nope_THN, k_rope_THR = k_THK.split([nope_head_dim, rope_head_dim], dim=-1)
    # MLA expands one rope vector per token onto every head; head 0 carries it.
    return k_nope_THN, k_rope_THR[:, 0]


def _expand_rope(k_nope_THN: torch.Tensor, k_rope_TR: torch.Tensor) -> torch.Tensor:
    """Rebuild the expanded key from its nope part and the rope slice."""
    k_rope_THR = k_rope_TR.unsqueeze(1).expand(-1, k_nope_THN.shape[1], -1)
    return torch.cat((k_nope_THN, k_rope_THR), dim=-1)


class MLAUlyssesCPFlexAttention(UlyssesCPFlexAttention):
    """Ulysses for MLA: one all-to-all for q, the nope key and v; the rope
    slice is all-gathered along the sequence and expanded on the local heads."""

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesCPFlexAttention.Config):
        rope_head_dim: int
        """Width of the rope slice at the end of the qk head dim."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.rope_head_dim = config.rope_head_dim

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = self.cp_group
        k_nope_THN, k_rope_TR = _split_rope(k_THK, self.rope_head_dim)
        widths = (q_THK.shape[-1], k_nope_THN.shape[-1], v_THV.shape[-1])
        # (T/cp, H, W) -> (T, H/cp, W) in one exchange.
        packed_THW = torch.cat((q_THK, k_nope_THN, v_THV), dim=-1)
        packed_THW = self._reshard(packed_THW, cp_group, src=_SEQ_DIM, dst=_HEAD_DIM)
        q_THK, k_nope_THN, v_THV = packed_THW.split(widths, dim=-1)
        # (T/cp, R) -> (T, R): the same vector on every head, so it never
        # travels expanded. Its backward is a reduce-scatter.
        k_rope_TR = spmd.redistribute(
            k_rope_TR.contiguous(),
            cp_group,
            src=spmd.S(_SEQ_DIM),
            dst=spmd.R,
            backward_options={"op_dtype": k_rope_TR.dtype},
        )
        out_THV = FlexAttention.forward(
            self, q_THK, _expand_rope(k_nope_THN, k_rope_TR), v_THV, **kwargs
        )
        # (T, H/cp, V) -> (T/cp, H, V).
        return self._reshard(out_THV, cp_group, src=_HEAD_DIM, dst=_SEQ_DIM)


class MLAAllGatherCPFlexAttention(AllGatherCPFlexAttention):
    """All-gather KV for MLA: the nope key and v travel packed, the rope slice
    travels once; q and the mask stay token-sharded."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllGatherCPFlexAttention.Config):
        rope_head_dim: int
        """Width of the rope slice at the end of the qk head dim."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.rope_head_dim = config.rope_head_dim

    def _gather(self, x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
        return spmd.redistribute(
            x,
            cp_group,
            src=spmd.S(_SEQ_DIM),
            dst=spmd.R,
            backward_options={"op_dtype": self.reduce_dtype or x.dtype},
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = self.cp_group
        k_nope_THN, k_rope_TR = _split_rope(k_THK, self.rope_head_dim)
        widths = (k_nope_THN.shape[-1], v_THV.shape[-1])
        packed_THW = self._gather(torch.cat((k_nope_THN, v_THV), dim=-1), cp_group)
        k_rope_TR = self._gather(k_rope_TR.contiguous(), cp_group)
        k_nope_THN, v_THV = packed_THW.split(widths, dim=-1)
        return FlexAttention.forward(
            self, q_THK, _expand_rope(k_nope_THN, k_rope_TR), v_THV, **kwargs
        )


def kcp_routing(
    seq_len_local: int,
    group: dist.ProcessGroup,
    *,
    conv_history: int,
    device: torch.device,
) -> ContextParallelRouting:
    """attn-gym routing for one sequence split into equal contiguous shards.

    Every rank owns ``[rank * L, (rank + 1) * L)`` of one document; CP rejects
    a load balancer so this table is the sharding the trainer actually
    applied. The plan is host-only; the routing is its device tensors, sized
    by the span and the conv's history.
    """
    world = dist.get_world_size(group)
    fragments = [[(r * seq_len_local, (r + 1) * seq_len_local)] for r in range(world)]
    plan = ContextParallelPlan.from_fragments(
        [0, seq_len_local * world], fragments, dist.get_rank(group)
    )
    return plan.routing(device, conv_history=conv_history)


class ContextParallelInnerKDA(ContextParallelKernel, InnerKDA):
    """Short convolution and KDA over one sequence shard per rank.

    The causal conv takes the previous rank's tail as history and the delta
    rule runs Attention Gym's context-parallel recipe, which exchanges
    per-fragment affine state summaries so each rank scans from its true
    entry state. The sequence stays sharded on both sides.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerKDA.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # Checked at build time so the message is actionable, rather than an
        # ImportError from inside a layer's first forward.
        try:
            from attn_gym.linear.context_parallel import (  # noqa: F401
                context_parallel_conv_history,
            )
            from attn_gym.linear.kda import context_parallel_kda  # noqa: F401
        except ImportError as err:
            raise ValueError(
                "KDA context parallelism needs attn-gym's context-parallel "
                "recipe (attn_gym.linear.kda.context_parallel_kda and "
                "attn_gym.linear.context_parallel.context_parallel_conv_history); "
                f"import failed with: {err}."
            ) from err

    def _routing(
        self,
        seq_len_local: int,
        group: dist.ProcessGroup,
        conv_history: int,
        device: torch.device,
    ) -> ContextParallelRouting:
        # The routing's tensors depend on the span and the history alone, so
        # one per shape serves every layer and step.
        key = (seq_len_local, dist.get_world_size(group), conv_history, str(device))
        cache = self.__dict__.setdefault("_routing_cache", {})
        if key not in cache:
            cache[key] = kcp_routing(
                seq_len_local, group, conv_history=conv_history, device=device
            )
        return cache[key]

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        from attn_gym.linear.context_parallel import context_parallel_conv_history

        if cu_seqlens is not None:
            raise NotImplementedError(
                "Kimi K3 KDA context parallel runs one document per batch; "
                "packed-document boundaries under CP are not supported yet."
            )
        group = self.cp_group
        mixed_qkv_1TC, conv_weight_C1W = self._pack_inputs(
            query_TC,
            key_TC,
            value_TC,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
        )
        seq_len_local = mixed_qkv_1TC.shape[1]
        routing = self._routing(
            seq_len_local, group, conv_weight_C1W.shape[-1] - 1, mixed_qkv_1TC.device
        )
        # One document per rank shard: the local span is one segment.
        cu_seqlens = torch.tensor(
            [0, seq_len_local], dtype=torch.int32, device=mixed_qkv_1TC.device
        )
        # The causal conv needs the previous rank's tail as history.
        conv_state = context_parallel_conv_history(mixed_qkv_1TC, routing, group)
        return self._conv_and_scan(
            mixed_qkv_1TC,
            conv_weight_C1W,
            raw_gate_THK,
            raw_beta_TH,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
            conv_state=conv_state,
            cp_routing=routing,
            cp_group=group,
        )
