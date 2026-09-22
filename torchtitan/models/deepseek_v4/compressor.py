# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module


@cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class Compressor(Module):
    """Compress local hidden states into lower-rate KV tokens.

    The compressor scores each token inside a compression group, forms a
    weighted sum, normalizes the result, and applies RoPE to the rope slice.
    For ``compress_ratio == 4`` it also includes the previous group's value as
    the overlapping candidate used by CSA.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wkv: Linear.Config
        wgate: Linear.Config
        norm: RMSNorm.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        compress_ratio: int = 4

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.overlap = cfg.compress_ratio == 4
        self.rope = cfg.rope.build()

        self.wkv = cfg.wkv.build()
        self.wgate = cfg.wgate.build()
        self.norm = cfg.norm.build()
        self.ape = nn.Parameter(torch.empty(cfg.compress_ratio, self.wkv.out_features))

    def _overlap_transform(self, tensor, value=0):
        """Append previous-token overlap candidates along the ratio dimension.

        Args:
            tensor: Grouped tensor of shape ``[B, L // R, R, D]``.
            value: Fill value for the first previous group.

        Returns:
            Tensor of shape ``[B, L // R, 2 * R, D]``.
        """
        d = self.head_dim
        prev = torch.cat(
            [
                torch.full_like(tensor[:1, :, :d], value),
                tensor[:-1, :, :d],
            ],
            dim=0,
        )
        curr = tensor[:, :, d:]
        return torch.cat([prev, curr], dim=1)

    def forward(self, x, positions):
        """Compress hidden states into compressed KV states.

        Args:
            x: Hidden states of shape ``[B, L, D_model]``.
            positions: Position IDs of shape ``[B, L]``.

        Returns:
            Compressed KV tensor of shape ``[B, L // compress_ratio, head_dim]``.
        """
        seqlen = x.size(0)
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv = self.wkv(x)
            score = self.wgate(x)
        if seqlen % ratio != 0:
            raise ValueError(
                f"seqlen ({seqlen}) must be divisible by compress_ratio ({ratio})"
            )
        comp_positions = positions[::ratio] if positions is not None else None
        kv = kv.unflatten(0, (-1, ratio))
        score = score.unflatten(0, (-1, ratio)) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0)
            score = self._overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(dtype))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = self.rope(kv_rope.unsqueeze(1), positions=comp_positions)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)
        return kv


class Indexer(Module):
    """Produce low-dimensional query/key features for CSA top-k selection."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wq_b: Linear.Config
        weights_proj: Linear.Config
        compressor: "Compressor.Config"
        num_index_heads: int = 64
        index_head_dim: int = 128
        rope_head_dim: int = 64

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.rope = cfg.rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        self.compressor = cfg.compressor.build()

    @staticmethod
    def _rotate_activation(x):
        dim = x.size(-1)
        hadamard_mat = _hadamard(dim, dtype=x.dtype, device=x.device)
        if isinstance(x, DTensor):
            hadamard_mat = DTensor.from_local(
                hadamard_mat,
                x.device_mesh,
                [Replicate()] * x.device_mesh.ndim,
                run_check=False,
            )
        return F.linear(x, hadamard_mat) * (dim**-0.5)

    def forward(
        self,
        x,
        qr,
        *,
        positions,
    ):
        """Project raw indexer queries, keys, and per-head weights."""
        seqlen = x.size(0)
        rd = self.rope_head_dim
        q = self.wq_b(qr)
        q = q.view(seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self.rope(q_rope, positions=positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = self._rotate_activation(q)
        k = self.compressor(x, positions=positions)
        k = self._rotate_activation(k)
        weights = self.weights_proj(x) * (
            self.softmax_scale * self.num_index_heads**-0.5
        )
        return q, k, weights

    @staticmethod
    def select(
        idx_q,
        idx_k,
        idx_w,
        *,
        seqlen: int,
        ratio: int,
        topk: int,
    ) -> torch.Tensor:
        """Select top-k compressed positions per folded query token.

        Selection is discrete, hence carries no gradient; the full ``[S, Hi, N]``
        score tensor never enters the autograd graph.  The student logits the
        indexer distillation loss trains are recomputed with gradient at the
        selected entries by :meth:`score_selected`.
        """
        with torch.no_grad():
            index_score = torch.einsum("shd,td->sht", idx_q, idx_k)
            index_score = index_score.relu_() * idx_w.unsqueeze(-1)
            index_score = index_score.sum(dim=1)

            compress_causal_limit = (
                torch.arange(1, seqlen + 1, device=idx_q.device).unsqueeze(1) // ratio
            )
            compress_causal_mask = (
                torch.arange(seqlen // ratio, device=idx_q.device).repeat(seqlen, 1)
                >= compress_causal_limit
            )
            index_score = index_score + torch.where(
                compress_causal_mask, torch.finfo(idx_q.dtype).min, 0
            )
            _, topk_indices = index_score.topk(min(topk, seqlen // ratio), dim=-1)
            return topk_indices

    @staticmethod
    def score_selected(
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return live indexer scores for the selected compressed positions.

        Unlike :meth:`select`, this runs inside the autograd graph: the returned
        logits are the student the distillation loss scores, so gradient flows
        back into the indexer projections.  Slots whose ``topk_indices`` are
        ``-1`` (not selectable) are marked ``-inf`` and dropped by the loss.
        """
        selected_TKD = idx_k[topk_indices.clamp_min(0)]
        logits_THK = torch.einsum("thd,tkd->thk", idx_q, selected_TKD)
        scores_TK = (logits_THK.relu() * idx_w.unsqueeze(-1)).sum(dim=1)
        return scores_TK.masked_fill(topk_indices < 0, -torch.inf)


class SparseIndexerLoss(AuxLoss):
    """Train sparse indexer scores from compressed-attention mass.

    The teacher is the attention probability mass on the compressed entries
    selected by the indexer, using the full sparse-attention softmax
    denominator supplied by FlexAttention's log-sum-exp. The selected-entry
    mass is intentionally not normalized before weighting: rows with little
    compressed attention mass contribute little loss.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """Aux-loss fields plus the attention softmax scale."""

        softmax_scale: float

    def __init__(self, config: Config):
        super().__init__(config)
        self.softmax_scale = config.softmax_scale

    def _teacher(
        self,
        q_THK: torch.Tensor,
        cmp_k_NK: torch.Tensor,
        topk_indices_TJ: torch.Tensor,
        lse_TH: torch.Tensor,
    ) -> torch.Tensor:
        """Return head-averaged attention mass on selected compressed entries."""
        valid_TJ = topk_indices_TJ >= 0
        row_valid_T = valid_TJ.any(dim=-1)

        selected_TJK = cmp_k_NK[topk_indices_TJ.clamp_min(0)]
        # In fp32: the mass ``exp(comp_lse - lse)`` is exponent-of-a-difference and
        # would lose precision (or underflow) if kept in bf16.
        logits_THJ = (
            torch.einsum("thk,tjk->thj", q_THK, selected_TJK).float()
            * self.softmax_scale
        )
        logits_THJ = logits_THJ.masked_fill(~valid_TJ.unsqueeze(1), -torch.inf)

        comp_lse_TH = torch.logsumexp(logits_THJ, dim=-1)
        mass_TH = torch.exp(comp_lse_TH - lse_TH.float())
        conditional_THJ = torch.softmax(
            logits_THJ.masked_fill(~row_valid_T[:, None, None], 0.0),
            dim=-1,
        )
        return (mass_TH.unsqueeze(-1) * conditional_THJ).sum(dim=1) / q_THK.size(1)

    def _logged_kl(
        self,
        p_TJ: torch.Tensor,
        t_TJ: torch.Tensor,
        logits_TJ: torch.Tensor,
        slot_valid_TJ: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-slot weighted KL values used for metrics and gradient."""
        log_student_TJ = F.log_softmax(
            logits_TJ.masked_fill(~slot_valid_TJ, -torch.inf).masked_fill(
                ~slot_valid_TJ.any(dim=-1, keepdim=True), 0.0
            ),
            dim=-1,
        )
        weighted_TJ = torch.special.xlogy(p_TJ, t_TJ) - p_TJ * log_student_TJ
        return weighted_TJ.masked_fill(~slot_valid_TJ, 0.0)

    def forward(
        self,
        q_THK: torch.Tensor,
        cmp_k_NK: torch.Tensor,
        topk_indices_TJ: torch.Tensor,
        lse_TH: torch.Tensor,
        topk_scores_TJ: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss and inject its gradient on ``carrier``.

        ``q_THK``, ``cmp_k_NK`` and ``lse_TH`` are teacher sources and should be
        detached by the caller. ``topk_scores_TJ`` is the live student input.
        """
        if topk_scores_TJ.numel() == 0:
            return carrier

        p_TJ = self._teacher(q_THK, cmp_k_NK, topk_indices_TJ, lse_TH)
        tiny = torch.finfo(torch.float32).tiny
        t_TJ = p_TJ / p_TJ.sum(dim=-1, keepdim=True).clamp_min(tiny)

        row_valid_T = torch.isfinite(topk_scores_TJ).any(dim=-1)
        logits_TJ = topk_scores_TJ.float().masked_fill(~row_valid_T.unsqueeze(-1), 0.0)
        slot_valid_TJ = torch.isfinite(topk_scores_TJ) & row_valid_T.unsqueeze(-1)
        raw_sum = self._logged_kl(p_TJ, t_TJ, logits_TJ, slot_valid_TJ).sum()
        return self.inject(raw_sum, carrier=carrier)
