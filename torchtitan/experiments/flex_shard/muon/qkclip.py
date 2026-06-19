# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""MuonClip (Kimi K2) QK-clip for DeepSeek V3 MLA attention."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.tools.logging import logger


def _is_mla_attention(module: nn.Module) -> bool:
    """A DeepSeek V3 MLA attention module (has the wkv_b proj + the head-dim split)."""
    return (
        hasattr(module, "wkv_b")
        and hasattr(module, "inner_attention")
        and hasattr(module, "n_heads")
        and hasattr(module, "qk_nope_head_dim")
        and hasattr(module, "qk_rope_head_dim")
    )


def _max_logit_per_head(
    q: torch.Tensor, k: torch.Tensor, scale: float, mask=None
) -> torch.Tensor:
    """Per-head max pre-softmax attention logit over the *attended* positions.

    ``q``/``k`` are ``[bsz, seqlen, n_heads, qk_head_dim]`` (the layout passed to the
    inner attention). Returns ``[n_heads]`` = max over batch and over the kept (i, j)
    positions of ``scale * (q_i . k_j)``. This is the "cheap separate pass" that
    materializes the score matrix the fused FlexAttention kernel does not expose.

    ``mask`` is the masking the real kernel applies, so the max is taken over exactly
    the positions the kernel attends to (not cross-document pairs it never sees, which
    would make ``tau`` a guarantee about the wrong score set). A FlexAttention
    ``BlockMask`` (causal + any document/packed masking) is materialized to a dense
    ``[B, 1, S, S]`` keep-mask via its ``mask_mod``; without one (e.g. plain SDPA) it
    falls back to a causal lower-triangular mask.
    """
    from torch.nn.attention.flex_attention import BlockMask, create_mask

    scores = torch.einsum("bihd,bjhd->bhij", q.float(), k.float()).mul_(scale)
    bsz, _, seqlen, _ = scores.shape
    if isinstance(mask, BlockMask):
        # Dense [B, 1, S, S] keep-mask from the kernel's own mask_mod (head-broadcast).
        keep = create_mask(mask.mask_mod, bsz, 1, seqlen, seqlen, device=scores.device)
    else:
        keep = torch.ones(
            seqlen, seqlen, dtype=torch.bool, device=scores.device
        ).tril_()
    scores.masked_fill_(~keep, float("-inf"))
    return scores.amax(dim=(0, 2, 3))


def _scale_rows(
    linear: nn.Module,
    row_scale: torch.Tensor,
    shard_map: dict[int, tuple] | None = None,
) -> None:
    """Scale output rows (dim 0) of a Linear's weight in place, sharding-aware.

    Reads the *persistent* parameter from ``module._parameters`` rather than the
    ``module.weight`` attribute: under FlexShard eager mode ``module.weight`` is a
    property backed by the bucket unshard hook and only materializes inside the module
    forward (it raises at optimizer-step time), whereas ``_parameters`` holds the
    persistent (owner-full / sharded) tensor the optimizer also operates on.

    Three placement regimes:

    * ``shard_map`` entry (gather methods -- ``Shard(0)`` / ``GroupedRaggedShard``): the
      weight is row- or byte-sharded across ranks. Build the full ``[out, in]`` scale
      matrix and extract this rank's local slice with
      :meth:`GatherMuon._local_update_shard` (the same full->local map the gather
      optimizer uses), then scale the local shard -- correct even when a byte cut splits
      a row across ranks.
    * ``Owned`` (no ``shard_map`` entry): the persistent weight is the full matrix on the
      owner (full row count) and empty on others -- scale all rows on the owner, skip
      empty shards.
    * Plain ``DTensor`` (row-sharded, no shard_map): skipped -- needs the global->local
      head-row mapping (fsdp2 QK-clip is a follow-up).

    DeepSeek V3 q/k projections have no bias (Linear ``bias=False``), so only the weight
    is scaled; a bias under sharding would need separate handling and raises.
    """
    from torch.distributed.tensor import DTensor

    weight = linear._parameters["weight"]
    bias = linear._parameters.get("bias")

    entry = shard_map.get(id(weight)) if shard_map is not None else None
    if entry is not None:
        placement, info, mesh = entry
        out_f, in_f = int(info.global_shape[0]), int(info.global_shape[1])
        if out_f != row_scale.shape[0]:
            return
        local = weight.data
        if local.numel() == 0:
            return
        if bias is not None:
            raise NotImplementedError("QK-clip on a sharded biased projection")
        full_scale = (
            row_scale.to(local.dtype).unsqueeze(1).expand(out_f, in_f).contiguous()
        )
        # Lazy import to avoid a qkclip <-> gather import cycle (gather imports QKClip).
        from .gather_muon import GatherMuon

        local_scale = GatherMuon._local_update_shard(
            full_scale, placement, info, mesh.get_local_rank(), mesh.size()
        )
        local.mul_(local_scale)
        return

    local = weight.to_local() if isinstance(weight, DTensor) else weight.data
    if local.shape[0] != row_scale.shape[0]:
        return
    local.mul_(row_scale.to(local.dtype).unsqueeze(1))
    if bias is not None:
        b_local = bias.to_local() if isinstance(bias, DTensor) else bias.data
        if b_local.shape[0] == row_scale.shape[0]:
            b_local.mul_(row_scale.to(b_local.dtype))


class QKClip:
    """MuonClip QK-clip (Kimi K2) for DeepSeek V3 MLA attention.

    After each optimizer step, rescales per-head query/key projection rows so the next
    forward's max attention logit is pulled back to ``tau``. The attention logit is
    bilinear in (Wq, Wk), so scaling both by ``gamma`` scales the logit by ``gamma``;
    with ``gamma_h = tau / S_max^h`` (only for heads whose observed max logit
    ``S_max^h > tau``, else 1) the head's max logit becomes ``tau``.

    For MLA only the *unshared* components are scaled (Kimi K2): the head-specific
    ``qC`` and ``kC`` (no-position) each by ``sqrt(gamma_h)`` and the head-specific
    ``qR`` (rotary query) by ``gamma_h``, while the *shared* rotary key ``kR`` (from
    ``wkv_a``) is left untouched to avoid coupling heads. So ``qC.kC`` scales by
    ``gamma_h`` and ``qR.kR`` scales by ``gamma_h`` -- both terms, hence the whole
    logit, scale by ``gamma_h``.

    ``S_max^h`` is captured each forward by wrapping the module's ``inner_attention``
    with an eager ``[B, H, S, S]`` score-materialization pass (:func:`_max_logit_per_head`,
    masked by the real ``BlockMask``); the wrapper stores it on the attention module, and
    :meth:`step` consumes the most recent value (reducing it across data-parallel ranks).
    The capture runs in the forward, so its cost is attributed to ``fwd_bwd`` in the
    benchmark, not ``opt_step`` (see :class:`bench._BenchMixin`); enabling QK-clip is the
    tax, independent of the dense-Muon distribution strategy.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        *,
        tau: float,
        shard_map: dict[int, tuple] | None = None,
    ) -> None:
        self.tau = float(tau)
        # shard_map: {id(persistent weight) -> (placement, info, mesh)} for gather
        # methods whose q/k projections are row-/byte-sharded; None for Owned (full on
        # owner). See :func:`_scale_rows`.
        self._shard_map = shard_map
        self._attns: list[nn.Module] = []
        for model in model_parts:
            for module in model.modules():
                if _is_mla_attention(module):
                    self._attns.append(module)
                    self._install_capture(module)

    @staticmethod
    def _install_capture(attn: nn.Module) -> None:
        inner = attn.inner_attention
        orig_forward = inner.forward

        def capture_forward(q, k, v, *args, **kwargs):
            scale = kwargs.get("scale", None)
            if scale is None:
                scale = attn.softmax_scale
            mask = kwargs.get("attention_masks", None)
            with torch.no_grad():
                attn._qkclip_smax = _max_logit_per_head(
                    q.detach(), k.detach(), scale, mask
                )
            return orig_forward(q, k, v, *args, **kwargs)

        inner.forward = capture_forward

    @torch.no_grad()
    def step(self) -> None:
        debug = os.environ.get("QKCLIP_DEBUG") == "1"
        max_logit = 0.0
        heads_clipped = 0
        for attn in self._attns:
            smax = getattr(attn, "_qkclip_smax", None)
            if smax is None:
                continue
            # Data-parallel correctness: each rank captured S_max from its own local
            # batch, so reduce to the global-batch per-head max before clipping --
            # otherwise the owner (Owned) clips on only its rank's logits, and sharded
            # rows would be scaled from different local maxima. MAX over the default
            # (world) group equals the data-parallel max when attention is not split by
            # head/sequence (tp=cp=1, the regime these configs run in); every rank runs
            # the full (replicated) attention on a distinct data slice, including under
            # EP (which partitions only experts).
            # TODO: with TP/CP, reduce only over the data-parallel axes -- heads/sequence
            # are partitioned there, so a plain world MAX would mix unrelated heads.
            if (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                dist.all_reduce(smax, op=dist.ReduceOp.MAX)
            if debug:
                max_logit = max(max_logit, float(smax.max()))
                heads_clipped += int((smax > self.tau).sum())
            gamma = torch.where(smax > self.tau, self.tau / smax, torch.ones_like(smax))
            if bool((gamma < 1.0).any()):
                self._rescale(attn, gamma)
        if debug:
            logger.info(
                f"[qkclip] pre-clip max_logit={max_logit:.2f} "
                f"heads_clipped={heads_clipped} tau={self.tau:.1f}"
            )

    def _rescale(self, attn: nn.Module, gamma: torch.Tensor) -> None:
        n_heads = attn.n_heads
        nope = attn.qk_nope_head_dim
        qk_head_dim = nope + attn.qk_rope_head_dim
        v_head_dim = attn.v_head_dim
        sqrt_gamma = gamma.sqrt()

        # Query projection rows per head: qC (nope) -> sqrt(gamma), qR (rope) -> gamma.
        qproj = attn.wq if attn.q_lora_rank == 0 else attn.wq_b
        q_scale = gamma.new_ones(n_heads * qk_head_dim)
        for h in range(n_heads):
            base = h * qk_head_dim
            q_scale[base : base + nope] = sqrt_gamma[h]
            q_scale[base + nope : base + qk_head_dim] = gamma[h]
        _scale_rows(qproj, q_scale, self._shard_map)

        # wkv_b rows per head: kC (nope) -> sqrt(gamma), v -> 1 (untouched). kR (shared
        # rotary key, from wkv_a) is never scaled.
        kvb = nope + v_head_dim
        kv_scale = gamma.new_ones(n_heads * kvb)
        for h in range(n_heads):
            base = h * kvb
            kv_scale[base : base + nope] = sqrt_gamma[h]
        _scale_rows(attn.wkv_b, kv_scale, self._shard_map)
