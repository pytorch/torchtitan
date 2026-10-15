# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CSA2 attention for DeepSeek V4.1.

Dense eager reference implementation -- no external kernel dependency.
Computes sliding-window + compressed-KV + attention-sink attention in a single
masked softmax. In training mode, computes the indexer distillation teacher
from the raw attention probability mass and injects its gradient via a
stop-gradient autograd function.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE

from .batched_linear import BatchedLinear
from .compressor import Compressor
from .context_parallel import QueryExchange
from .indexer import HierarchicalIndexer


class _AttachIndexerLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, loss):
        ctx.save_for_backward(loss)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(loss)


def indexer_objective(scores_BLK, teacher_BLK, valid_BLK, num_queries):
    """Selected-support cross entropy with raw, detached attention mass.

    Its logit gradient is (Z * softmax(scores) - p) / num_queries, where
    Z is p.sum(-1). Do not renormalize p: SWA and sink consume part of the
    attention probability mass. Rows without a completed group contribute zero.
    """
    safe_BLK = scores_BLK.float().masked_fill(~valid_BLK, -torch.inf)
    safe_BLK = torch.where(valid_BLK.any(-1, keepdim=True), safe_BLK, 0)
    teacher_BLK = teacher_BLK.detach().float().masked_fill(~valid_BLK, 0)
    finite_BLK = scores_BLK.float().masked_fill(~valid_BLK, 0)
    loss_BL = teacher_BLK.sum(-1) * torch.logsumexp(safe_BLK, -1)
    loss_BL -= (teacher_BLK * finite_BLK).sum(-1)
    return loss_BL.sum() / num_queries


class CompressedSparseInnerAttention2(Module):
    """CSA2's sparse core: sliding window plus the selected compressed entries.

    The whole core is Attention Gym's ``gather_attn`` in one call: ``q`` carries
    ``H`` heads while both KV sources carry a single shared head, and ``topk_indices``
    index the compressed KV -- inside each document's own entries -- with ``-1`` for unused
    slots.  ``compress_ratio`` carries the layer's verified ratio (0 window-only, 1 a
    full-resolution second KV stream, 2 the compressed shared KV): it is the grid the
    sparse pool is addressed on -- a fused port also needs it to translate the metadata's
    document boundaries into its kernel layout -- so it is never guessed from shapes.

    A port that can produce the indexer's teacher replaces this forward wholesale and
    emits it on ``topk_scores``' gradient; this core has no kernel to make one.  Which port
    is in place is read off the requested overrides, one stage earlier, by
    :func:`torchtitan.models.deepseek_v4_1.model._check_indexer_teacher_pair`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        softmax_scale: float
        # The layer's compression ratio, carried explicitly for the fused ports.
        compress_ratio: int

    def __init__(self, config: Config):
        super().__init__()
        self.window_size = config.window_size
        self.softmax_scale = config.softmax_scale
        self.compress_ratio = config.compress_ratio
        self.num_global_queries = None
        self.last_indexer_loss = None

    def forward(
        self,
        q: torch.Tensor,
        swa_k: torch.Tensor,
        attn_sink: torch.Tensor,
        attention_masks,
        *,
        cmp_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense eager CSA2 with a stop-gradient indexer teacher.

        Computes sliding-window attention, compressed-KV attention, and the
        attention sink in a single masked softmax. In training mode, computes
        the indexer distillation teacher from the raw attention probability
        mass and injects its gradient via ``_AttachIndexerLoss``.

        Args:
            q: ``[B, L, H, Dk]``
            swa_k: ``[B, L, Dk]``
            attn_sink: ``[H]``
            attention_masks: ``DeepSeekV41Metadata`` with document boundaries.
            cmp_k: ``[B, N, Dk]`` or None.
            topk_indices: ``[B, L, K]`` (-1 for unused) or None.
            topk_scores: ``[B, L, K]`` student logits or None.
        """
        if q.shape[0] != 1:
            raise ValueError("ReferenceCSA2 requires a single packed row (batch=1)")
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together.")
        length = q.shape[1]
        positions_L = torch.arange(length, device=q.device)
        boundaries = attention_masks.q.cu_seqlens
        documents_L = torch.searchsorted(boundaries[1:], positions_L, right=True)
        starts_L = boundaries[documents_L]
        keys_S = positions_L
        visible_LS = keys_S[None, :] >= starts_L[:, None]
        visible_LS &= keys_S[None, :] <= positions_L[:, None]
        visible_LS &= keys_S[None, :] > positions_L[:, None] - self.window_size
        logits_BLHS = torch.einsum("blhd,bsd->blhs", q.float(), swa_k.float())
        logits_BLHS = (logits_BLHS * self.softmax_scale).masked_fill(
            ~visible_LS[None, :, None, :], -torch.inf
        )
        selected_BLKD = None
        if cmp_k is not None:
            if topk_indices is None or topk_scores is None:
                raise ValueError("Compressed KV requires indices and student scores")
            frame = attention_masks.frame_for(self.compress_ratio)
            offsets_L = frame.cu_seqlens[documents_L]
            indices_LK = topk_indices.squeeze(0) + offsets_L[:, None]
            selected_BLKD = cmp_k[:, indices_LK.clamp_min(0)]
            logits_BLHK = torch.einsum(
                "blhd,blkd->blhk", q.float(), selected_BLKD.float()
            )
            logits_BLHK = (logits_BLHK * self.softmax_scale).masked_fill(
                (topk_indices < 0)[:, :, None, :], -torch.inf
            )
            logits_BLHS = torch.cat((logits_BLHS, logits_BLHK), -1)
        sink_BLH1 = attn_sink.float().view(1, 1, -1, 1).expand(1, length, -1, 1)
        probabilities = torch.cat((logits_BLHS, sink_BLH1), -1).softmax(-1)
        output_BLHD = torch.einsum(
            "blhs,bsd->blhd", probabilities[..., :length], swa_k.float()
        )
        if selected_BLKD is not None:
            sparse_probs_BLHK = probabilities[..., length:-1]
            output_BLHD += torch.einsum(
                "blhk,blkd->blhd", sparse_probs_BLHK, selected_BLKD.float()
            )
            if self.training and self.num_global_queries is not None:
                teacher_BLK = sparse_probs_BLHK.detach().sum(2)
                loss = indexer_objective(
                    topk_scores, teacher_BLK, topk_indices >= 0, self.num_global_queries
                )
                self.last_indexer_loss = loss.detach()
                output_BLHD = _AttachIndexerLoss.apply(output_BLHD, loss)
        return output_BLHD.to(q.dtype)


class Attention(BaseAttention):
    """Latent attention with a grouped output projection, CSA2's per-layer wrapper.

    Projections, both RoPE phases, the compressor and the indexer live here; the sparse
    core is the ``inner_attention`` module.  The shared cross-layer tensors (compressed
    KV, index keys, selected entries, student logits, candidate pool) are threaded
    through as ordinary inputs and returned updated, so a layer's role is visible at the
    call site rather than hidden in mutable state.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        head_dim: int
        # The pinned V3 checkpoint mapping branches on ``q_lora_rank`` (it selects the
        # LoRA-style HF keys), so the field stays even though this module only needs the
        # projection configs.
        q_lora_rank: int
        compress_ratio: int
        inner_attention: CompressedSparseInnerAttention2.Config  # pyrefly: ignore [bad-override]
        rope: RoPE.Config
        compressor: Compressor.Config
        indexer: HierarchicalIndexer.Config
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config
        # The query side's context-parallel movement; it is a module only because the process
        # group it runs on is run-level state, wired by ``parallelize``.
        query_exchange: QueryExchange.Config = field(default_factory=QueryExchange.Config)

    def __init__(self, config: "Attention.Config"):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.compress_ratio = cfg.compress_ratio
        self.rope = cfg.rope.build()
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        # One sink logit per head, fp32 as in the released checkpoint and the kernels.
        self.attn_sink = nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))
        self.compressor = cfg.compressor.build()
        self.indexer = cfg.indexer.build()
        self.inner_attention = cfg.inner_attention.build()
        self.query_exchange = cfg.query_exchange.build()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_masks,
        *,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(o, cmp_k, idx_k, topk_indices, topk_scores, candidates)``.

        The returned shared tensors are this layer's contribution to the chain: freshly
        computed where the layer is a source, otherwise the inputs unchanged.
        """
        bsz, seqlen, _ = x.size()
        # Under context parallelism the queries are this rank's chunks while the keys stay the
        # slab it holds: the query side is a permuted copy of the input, and only the
        # projections move -- never the hidden states -- so the keys keep the rows and the
        # positions they always had, and the queries rotate at the ones the metadata carries.
        x_q = self.query_exchange.permute(x.squeeze(0), attention_masks.exchange).unsqueeze(0)

        qr = self.q_norm(self.wq_a(x_q))
        q = self.wq_b(qr).view(bsz, seqlen, -1, self.head_dim)
        swa_k = self.kv_norm(self.wkv(x))

        # Every layer compresses and indexes: a source publishes new tensors, a reusing
        # layer hands back the ones in flight (asserted inside those modules).
        cmp_k, latent = self.compressor(x, positions, cmp_k)
        # The indexer is trained by distillation alone, so it reads the trunk as
        # constants.  The shared index keys are the exception: a Reindex Mode layer keeps
        # the ``idx_k`` it was handed live, so its consumers go on training the key's
        # owner.
        idx_k, topk_indices, topk_scores, candidates = self.indexer(
            x_q.detach(),
            qr.detach(),
            positions,
            attention_masks,
            latent=latent.detach() if latent is not None else None,
            idx_k=idx_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            candidates=candidates,
        )

        # The rope config carries the un-rotated prefix width, so the module rotates the
        # trailing span and puts the prefix back.
        q = self.rope(q, positions=attention_masks.positions_q)
        # The shared KV latent is one rank-2 head; RoPE rotates rank-3 [B, L, N, H].
        swa_k = self.rope(swa_k.unsqueeze(2), positions=positions).squeeze(2)

        uses_cmp = self.compress_ratio > 0
        o = self.inner_attention(
            q,
            swa_k,
            self.attn_sink,
            attention_masks,
            cmp_k=cmp_k if uses_cmp else None,
            topk_indices=topk_indices if uses_cmp else None,
            topk_scores=topk_scores if uses_cmp else None,
        )
        # Rotate back and project while the rows are still this rank's chunks: every step here is
        # per token, so it commutes with the movement, and ``wo_b``'s output is an order of
        # magnitude narrower than the attention's -- so that is what takes the collective.
        o = self.rope(o, positions=attention_masks.positions_q, inverse=True)

        # The output projection is grouped: wo_a projects each query-head group on its
        # own, wo_b mixes the per-group results back to the model dimension.  The group
        # count comes from the module so a group-wise sharding can narrow it later.
        o = self.wo_a(o.reshape(bsz * seqlen, self.wo_a.n_batches, -1))
        o = self.wo_b(o.flatten(-2))
        # A permutation, so undoing it is exact and accumulates nothing; ``wo_b`` has already
        # flattened the batch axis, so the row axis is dim 0 here.  Unsharded, the exchange has
        # no plan and hands the tensor straight back.
        o = self.query_exchange.unpermute(o, attention_masks.exchange)
        return (o, cmp_k, idx_k, topk_indices, topk_scores, candidates)
