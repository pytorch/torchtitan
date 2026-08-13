# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model components that fold the TP collectives into their GEMMs.

:class:`AllGatherFusedQKVLinear`, :class:`RowParallelLinear` and
:class:`AllGatherFusedFeedForward` are drop-in replacements for the stock QKV,
output and SwiGLU projections. They keep the stock parameter layouts and only
move the TP collective into the GEMM, over the autograd Functions in
``torchtitan/distributed/linear.py``. ``RowParallelLinear`` serves both attention's ``wo`` and the
FFN's ``w2``; nothing about the primitives is attention-specific, and MoE
projections could use the same pair.

What lives here is the wiring -- the reshaping around each collective, and the
fallbacks -- while ``torchtitan/distributed/linear.py`` holds the collective+GEMM math itself.

Selected by passing ``tp_comm_overlap="dist_gemm"`` to ``make_gqa_config`` or
``make_ffn_config`` (see ``config_utils.py``), which also drops the boundary
all-gather these modules take over. No attention or FFN subclass is needed beyond
the projections: the stock ``GQAttention`` forward handles a QKV that changes the
sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchtitan.distributed.linear import (
    AllGatherLinear,
    AllGatherLinearMulti,
    dist_gemm_workspace_bytes,
    LinearReduceScatter,
    reserve_symm_mem_workspace,
)

from torchtitan.distributed.spmd_types import current_spmd_mesh

from torchtitan.models.common.attention import FusedQKVLinear, GQAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear

if TYPE_CHECKING:

    from torchtitan.distributed.parallel_dims import ParallelDims


def _tp_group_from_context() -> dist.ProcessGroup | None:
    """The TP process group from the current spmd_types mesh context, or None.

    Resolved per forward rather than captured at parallelize time. The mesh
    context is only entered inside the trainer's ``train_context``, so it is
    unavailable during ``__init__`` and ``parallelize`` -- and reading it here
    means these modules need no ``parallelize`` override and hold no group state.

    None means "run the stock projection": either no mesh context (non-spmd_types
    caller) or TP is degree 1, in which case there is no collective to fuse.
    """
    mesh = current_spmd_mesh()
    if mesh is None or "tp" not in (mesh.mesh_dim_names or ()):
        return None
    tp_group = mesh.get_group("tp")
    return tp_group if tp_group.size() > 1 else None


class _ReservesSymmMemWorkspace:
    """Mixin: size the symmetric-memory workspace before this layer's first ops.

    ``tokens_per_rank`` is the per-step token count, which only the runtime config
    knows, so ``maybe_update_dist_gemm_config`` stamps it onto the module configs
    at ``update_from_config`` time and ``__init__`` stores it. Same shape as
    ``DeepEPTokenDispatcher``, which takes ``num_max_tokens_per_rank`` from its
    config and defers the buffer allocation.

    The reservation itself cannot happen in ``__init__``: sizing the workspace
    needs the TP process group, and the mesh does not exist until well after
    ``build()`` -- which is why the referenced example defers its allocation too.
    It cannot happen in ``forward`` either: allocating and rendezvousing is not
    traceable, so it would break ``--compile.enable`` and whole-model tracing.

    So it hangs off ``Module.parallelize``, which is the generic hook every trainer
    calls (``model.parallelize(parallel_dims)``) rather than a model-specific
    ``parallelize_fn``. That matters: a trainer substituting its own
    ``parallelize_fn`` -- GraphTrainer does -- would silently skip a hook placed
    there, leaving the workspace to grow lazily.

    ``tokens_per_rank`` is None for callers that never saw a runtime config
    (inference, unit tests). The ops then size the workspace lazily as they always
    have -- correct, just without the pre-capture guarantee.
    """

    tokens_per_rank: int | None

    def _init_workspace_reservation(self, tokens_per_rank: int | None) -> None:
        self.tokens_per_rank = tokens_per_rank

    def _workspace_weights(self) -> tuple[torch.Tensor, ...]:
        """The sharded weights whose dims bound this layer's fused GEMMs."""
        raise NotImplementedError

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        # A mixin, so the static view of super() is object; every concrete user
        # mixes this in ahead of a Module subclass that does define parallelize.
        super().parallelize(parallel_dims)  # pyrefly: ignore[missing-attribute]
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is None or self.tokens_per_rank is None:
            return
        tp_group = tp_mesh.get_group("tp")
        if tp_group.size() < 2:
            return
        # After super(), so the weights are sharded and both of their dims are
        # local; the widest bounds the K and N any of this layer's GEMMs will see.
        weights = self._workspace_weights()
        reserve_symm_mem_workspace(
            tp_group,
            min_bytes=dist_gemm_workspace_bytes(
                tokens_global=self.tokens_per_rank,
                features=max(dim for w in weights for dim in w.shape),
                ranks=tp_group.size(),
            ),
        )


def maybe_update_dist_gemm_config(model_config: object, config: object) -> None:
    """Stamp the per-step token count onto any dist-GEMM module configs.

    Mirrors ``update_ep_token_dispatcher_config``. Also the one place that sees
    both the model config and the runtime parallelism config, so the two
    preconditions are checked here: neither is detectable from inside a module,
    because under spmd_types an activation is a plain local tensor with no
    placements to inspect.
    """
    cfgs: list[object] = []
    for layer_cfg in getattr(model_config, "layers", []):
        attn_cfg = getattr(layer_cfg, "attention", None)
        if isinstance(attn_cfg, GQAttention.Config) and isinstance(
            attn_cfg.wo, RowParallelLinear.Config
        ):
            cfgs.extend((attn_cfg.qkv_linear, attn_cfg.wo))
        ffn_cfg = getattr(layer_cfg, "feed_forward", None)
        if isinstance(ffn_cfg, AllGatherFusedFeedForward.Config):
            cfgs.append(ffn_cfg)
    if not cfgs:
        return

    from torchtitan.trainer import Trainer

    if not isinstance(config, Trainer.Config):
        # Inference-only callers have no fixed token count per step.
        return

    parallelism = config.parallelism
    if parallelism.spmd_backend != "spmd_types":
        raise ValueError(
            "tp_comm_overlap='dist_gemm' requires "
            "parallelism.spmd_backend='spmd_types', got "
            f"{parallelism.spmd_backend!r}. The fused modules take and return plain "
            "local tensors; the DTensor backends are being deprecated and are not "
            "supported."
        )
    if not parallelism.enable_sequence_parallel:
        raise ValueError(
            "tp_comm_overlap='dist_gemm' requires "
            "parallelism.enable_sequence_parallel; the fused GEMMs replace the SP "
            "all-gather and reduce-scatter, so there is nothing for them to fuse "
            "with SP disabled."
        )

    tokens_per_rank = config.training.local_batch_size * config.training.seq_len
    for cfg in cfgs:
        cfg.tokens_per_rank = tokens_per_rank  # pyrefly: ignore[missing-attribute]


class AllGatherFusedQKVLinear(_ReservesSymmMemWorkspace, FusedQKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(FusedQKVLinear.Config):
        """Same fields as the stock fused QKV, plus the token count the workspace
        reservation needs. The subclass also binds ``Config.build()`` to this
        module rather than the stock one, so it cannot be deleted as empty."""

        tokens_per_rank: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self._init_workspace_reservation(config.tokens_per_rank)

    def _workspace_weights(self) -> tuple[torch.Tensor, ...]:
        return (self.wqkv.weight,)

    def forward(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            return super().forward(x)

        bsz, _, dim = x.shape
        # The all-gather concatenates whole per-rank blocks, so the rows it
        # produces are ordered (rank, batch, seq_local). Flattening [B, S/W, D]
        # directly would therefore be reinterpreted as (batch, seq) and mix
        # batches together for bsz > 1. Put the sequence outermost first so the
        # gathered rows really are (seq, batch) row-major.
        x_seq_major = x.transpose(0, 1).reshape(-1, dim).contiguous()
        qkv_flat = AllGatherLinear.apply(
            x_seq_major,
            self.wqkv.weight,
            self.wqkv.bias,
            tp_group,
            tp_group.group_name,
        )

        full_seqlen = qkv_flat.shape[0] // bsz
        qkv = qkv_flat.view(
            full_seqlen,
            bsz,
            -1,
            self.r_dim,
            self.head_dim,
        ).transpose(0, 1)
        xq, xk, xv = torch.split(qkv, [self.heads_per_kv, 1, 1], dim=-2)
        return (
            xq.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous(),
            xk.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous(),
            xv.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous(),
        )


class RowParallelLinear(_ReservesSymmMemWorkspace, Linear):
    """Attention output projection: matmul fused with the TP reduce-scatter.

    Named for the role it fills rather than the collective it performs, so it does
    not read like the :class:`LinearReduceScatter` autograd Function it calls. The
    class itself is a plain rowwise linear and would work for any row-parallel
    projection; today it is only wired in as ``wo``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Same fields as a stock Linear, plus the token count the workspace
        reservation needs. The subclass also binds ``Config.build()`` to this
        module rather than the stock one, so it cannot be deleted as empty."""

        tokens_per_rank: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self._init_workspace_reservation(config.tokens_per_rank)

    def _workspace_weights(self) -> tuple[torch.Tensor, ...]:
        return (self.weight,)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            return super().forward(input)

        bsz, seqlen, k_local = input.shape
        world_size = tp_group.size()
        # Reduce-scatter splits the flattened rows, so put the sequence outermost
        # first or the split would cut across batches instead of the sequence.
        # Feeding 2D with scatter_dim=0 is also what lets the operator take its
        # fused schedules, which it declines for a 3D input.
        x_seq_major = input.transpose(0, 1).reshape(-1, k_local).contiguous()
        y_flat = LinearReduceScatter.apply(
            x_seq_major,
            self.weight,
            self.bias,
            tp_group,
            tp_group.group_name,
        )
        return y_flat.view(seqlen // world_size, bsz, -1).transpose(0, 1).contiguous()


class AllGatherFusedFeedForward(_ReservesSymmMemWorkspace, FeedForward):
    """SwiGLU feed-forward with both TP collectives folded into its GEMMs.

    ``w1`` and ``w3`` share an input, so one all-gather feeds both
    (:class:`AllGatherLinearMulti`); ``w2`` is row-parallel and reduce-scatters
    back to a sequence shard (:class:`LinearReduceScatter`). Parameter layout and
    checkpoint FQNs are the stock ``w1``/``w2``/``w3``.

    Falls back to the stock forward when TP is off, or when ``w1``/``w3`` carry a
    bias: the multi-weight gather takes no per-weight bias (torchtitan's dense FFN
    builds these with ``bias=False``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        """Same fields as the stock FFN, plus the token count the workspace
        reservation needs. The subclass also binds ``Config.build()`` to this
        module rather than the stock one, so it cannot be deleted as empty."""

        tokens_per_rank: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self._init_workspace_reservation(config.tokens_per_rank)

    def _workspace_weights(self) -> tuple[torch.Tensor, ...]:
        return (self.w1.weight, self.w2.weight, self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None or self.w1.bias is not None or self.w3.bias is not None:
            return super().forward(x)

        bsz, _, dim = x.shape
        # Sequence outermost before flattening, so the gathered rows really are
        # (seq, batch) row-major. See AllGatherFusedQKVLinear for why.
        x_seq_major = x.transpose(0, 1).reshape(-1, dim).contiguous()
        h1, h3 = AllGatherLinearMulti.apply(
            x_seq_major,
            self.w1.weight,
            self.w3.weight,
            tp_group,
            tp_group.group_name,
        )
        # Elementwise on feature-sharded activations: no collective.
        h = F.silu(h1) * h3
        y_flat = LinearReduceScatter.apply(
            h,
            self.w2.weight,
            self.w2.bias,
            tp_group,
            tp_group.group_name,
        )
        # y_flat is [S_local * B, dim], sequence-major. Shape comes from y_flat
        # rather than from x: the collectives change the row count.
        return y_flat.view(-1, bsz, y_flat.shape[-1]).transpose(0, 1).contiguous()


__all__ = [
    "AllGatherFusedFeedForward",
    "AllGatherFusedQKVLinear",
    "RowParallelLinear",
    "maybe_update_dist_gemm_config",
]
