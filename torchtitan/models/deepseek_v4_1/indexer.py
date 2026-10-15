# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning indexer for DeepSeek V4.1 (CSA2).

Shape legend for this file:
    B = batch, L = sequence length, D = model dimension,
    Hi = ``num_index_heads``, Di = ``index_head_dim``,
    N = number of compressed KV entries (``L // compress_ratio``),
    K = ``index_topk`` (entries selected per query).

Score of query ``t`` against compressed entry ``j``::

    S_{t,h,j} = <q^I_{t,h}, k^I_j>
    I_{t,j}   = sum_h w_{t,h} * relu(S_{t,h,j})

The top-``K`` entries of ``I_{t,.}`` are what the sparse attention reads.  Selection is
discrete, hence carries no gradient.

The distillation objective that trains the indexer is deliberately **not** in this file.
It needs the attention's own softmax denominator (window + selected compressed + sink), so
it belongs to the attention that produces it, not to the selector; the fused port carries
it out through ``topk_scores``' gradient, on the edge SLIKG consumes.  Nothing here holds
a reference to a loss.

The whole computation is single-pass: no query chunking, which is a kernel-side concern
and not something the reference implementation should carry.  A kernel-backed variant
replaces :class:`Selector`'s forward: the per-head weights and the relaxed score, the
visibility and candidate masking, and the top-k selection are what a fused
``lightning_indexer`` kernel would implement.

Packed documents are addressed the way Attention Gym's ``gather_attn`` addresses them, one
axis over: the metadata's packed offsets say which entries a query's own document
contributed, and the query may select only the groups that are complete at it.  Entry ``j``
covers tokens ``[j * compress_ratio, (j + 1) * compress_ratio)`` of its document, so a
query reads the candidates ``[start, end)`` of the whole-row pool where ``start`` is its own
document's pool offset and ``end`` is ``start`` plus the number of groups its own positions
complete.  The selection is emitted inside that interval -- the document-local numbering
``gather_attn`` and the fused kernels both read -- so no consumer translates it.

The interval arithmetic is exact only when every document segment is a multiple of
``compress_ratio`` tokens; otherwise a pooling group straddles a document edge and its
entry mixes the two documents.  The loaders pad every document up to
:func:`~torchtitan.models.deepseek_v4_1.model.compression_alignment`, so the
straddling group does not arise in the packed layout they produce.

Every layer owns a :class:`HierarchicalIndexer`, statically assigned one of CSA2's three
modes: Full Mode carries the parameters and produces both the index keys and the top-k,
Reindex Mode carries its own query and rescores the shared keys, and Reuse Mode carries
the shared top-k forward without computing anything.  The mode is asserted in ``forward``,
so a misconfigured layer fails loudly instead of silently recomputing or silently reusing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torchtitan.models.base.module import Module

if TYPE_CHECKING:
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.common.nn_modules import RMSNorm
    from torchtitan.models.common.rope import RoPE

    from .model import DeepSeekV41Metadata


class IndexerMode(StrEnum):
    """CSA2's static per-layer modes (report section 2.3.1).

    ``FULL`` owns the main KV and projects its own index keys, ``REINDEX`` rescores the
    shared keys with its own query, and ``REUSE`` computes no index query at all.
    """

    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


# Bound at module level so the mode adapters in HierarchicalIndexer read as
# "if self.mode is REUSE".
FULL = IndexerMode.FULL
REINDEX = IndexerMode.REINDEX
REUSE = IndexerMode.REUSE


def _document_intervals(
    cu_seqlens: torch.Tensor, cu_seqlens_k: torch.Tensor, *, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(pool_start_L1, pool_end_L1, local_L1)``: each query's own document's entries, and the
    query's place in it.  ``pool_start`` is both the origin of a query's reach and the shift
    between the row-wide pool numbering and the selection's per-document one.
    """
    positions = torch.arange(num_tokens, device=cu_seqlens.device)
    documents = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    pool_start_L1 = cu_seqlens_k.index_select(0, documents).unsqueeze(-1)
    pool_end_L1 = cu_seqlens_k.index_select(0, documents + 1).unsqueeze(-1)
    local_L1 = (positions - cu_seqlens.index_select(0, documents)).unsqueeze(-1)
    return pool_start_L1, pool_end_L1, local_L1


def _selection_mask(
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    num_tokens: int,
    num_cmp: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Document isolation plus causal completeness over the compressed-entry axis.

    Returns ``(visible_LN, newest_L1, newest_valid_L1, pool_start_L1)``, in the row-wide pool
    numbering the scores live in.  A group counts once the query's own document has produced
    it, so the causal limit is ``pool_start + (local + 1) // compress_ratio``.
    ``newest_valid_L1`` says whether there is a complete group at all, which is what the
    candidate pool pins a block for.
    """
    pool_start_L1, pool_end_L1, local_L1 = _document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=num_tokens)
    ends_L1 = torch.minimum(pool_end_L1, pool_start_L1 + (local_L1 + 1) // compress_ratio)

    entry_1N = torch.arange(num_cmp, device=cu_seqlens.device).unsqueeze(0)
    visible_LN = (entry_1N >= pool_start_L1) & (entry_1N < ends_L1)
    return visible_LN, ends_L1 - 1, ends_L1 > pool_start_L1, pool_start_L1


def _student_logits(
    idx_q_BLHiDi: torch.Tensor,
    idx_k_BNDi: torch.Tensor,
    weights_BLHi: torch.Tensor,
    local_BLK: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor:
    """The per-head scores at a document-local selection: the pass that carries gradient.

    Selection is discrete, so only the picks can be scored differentiably -- and an
    indices-only selection leaves nothing to gather them from, which is why this is a pass of
    its own rather than a gather out of the selection pass.  ``local_BLK`` is renumbered onto
    the row-wide entries the keys live in; a ``-1`` slot is gathered harmlessly and marked
    ``-inf``.  The row is the batch, so the key axis is squeezed.  Returns ``[B, L, K]``,
    head-summed.
    """
    pool_start_L1, _, _ = _document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=idx_q_BLHiDi.size(1))
    # A ``-1`` slot may name a real entry once shifted; the value is discarded below.
    selected_BLKDi = idx_k_BNDi.squeeze(0)[(local_BLK + pool_start_L1).squeeze(0).clamp_min(0)].unsqueeze(0)
    logits_BLHiK = torch.einsum("blhd,blkd->blhk", idx_q_BLHiDi, selected_BLKDi)
    logits_BLHiK = logits_BLHiK.relu() * weights_BLHi.unsqueeze(-1)
    return logits_BLHiK.sum(dim=2).masked_fill(local_BLK < 0, -torch.inf)


class Selector(Module):
    """The selection node: scores every visible compressed entry and picks the top-k.

    The pool is scored row-wide -- one axis over the packed documents -- while the picks are
    numbered inside each query's own document, which is the numbering the sparse attention
    and the fused kernels read.

    It is its own module so a fused kernel can replace ``forward`` without replacing the
    whole indexer -- the indexer's projections, norms and rope stay where they are.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mode: IndexerMode
        compress_ratio: int
        num_index_heads: int
        index_head_dim: int
        index_topk: int
        # Candidate pool, in the kernels' own spelling: a negative capacity scores every
        # visible entry, a positive one (with a block size of ``CANDIDATE_BLOCK_SIZE``)
        # builds or searches the shared pool.  Which of those two a capacity-carrying layer
        # does is ``mode``'s answer, not the config's: only a Full Mode layer owns the
        # compressed KV the pool indexes, so only it can build one.
        candidate_topk_blocks: int = -1
        candidate_block_size: int = -1

    def __init__(self, config: Config):
        super().__init__()
        self.num_global_queries: torch.Tensor | None = None
        self.mode = config.mode
        self.compress_ratio = config.compress_ratio
        self.num_index_heads = config.num_index_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size

    @property
    def has_candidate_pool(self) -> bool:
        """Whether this layer builds or searches the shared candidate pool.

        The test is the kernel's own: it enables the pool on a positive capacity and rejects
        anything else, so a value that is neither positive nor the disabled sentinel is the
        operator's to refuse, not this property's to guess at.
        """
        return self.candidate_topk_blocks > 0

    @staticmethod
    def select_candidate_blocks(
        scores_BLN: torch.Tensor,
        newest_L1: torch.Tensor,
        newest_valid_L1: torch.Tensor,
        topk_blocks: int,
        block_size: int,
    ) -> torch.Tensor:
        """The candidate source's blockwise selection: keep the best-scoring blocks.

        Args:
            scores_BLN: Index scores ``[B, L, N]``, already masked to ``-inf`` on entries
                the query cannot select (other documents and incomplete groups).
            newest_L1: Index of each query's newest selectable entry, ``[L, 1]``.
            newest_valid_L1: Whether that entry exists (the query's document has at least
                one complete group), ``[L, 1]``.
            topk_blocks: Maximum number of blocks to keep.
            block_size: Positions per block.

        Returns:
            Boolean mask ``[B, L, N]`` selecting every position of the kept blocks.

        The two entry-index tables describe the whole packed row, so they carry no batch axis
        and broadcast against ``scores_BLN``: a query's reach is an interval of the row-wide
        pool, and its newest group is what pins a block.
        """
        width = scores_BLN.size(-1)
        if width % block_size != 0:
            scores_BLN = F.pad(scores_BLN, (0, -width % block_size), value=-torch.inf)
        # A block is scored by its best position, which is what makes the pool
        # recall-oriented rather than a second token-level selection.
        block_scores_BLB = scores_BLN.unflatten(-1, (-1, block_size)).amax(dim=-1)
        num_blocks = block_scores_BLB.size(-1)

        # The block holding a query's newest selectable entry is only partly filled, and
        # must not be outscored by an older, full block.  A query whose document has no
        # complete group yet owns no such block and pins nothing.
        last_L1 = newest_L1 // block_size
        pin_LB = torch.arange(num_blocks, device=scores_BLN.device).unsqueeze(0) == last_L1
        block_scores_BLB = block_scores_BLB.masked_fill(pin_LB & newest_valid_L1, torch.inf)

        top = block_scores_BLB.topk(min(topk_blocks, num_blocks), dim=-1)
        # Fewer reachable blocks than ``topk_blocks`` leaves -inf picks behind: drop them.
        keep_BLB = torch.zeros_like(block_scores_BLB, dtype=torch.bool).scatter_(
            -1, top.indices, top.values > -torch.inf
        )
        return keep_BLB.repeat_interleave(block_size, dim=-1)[..., :width]

    def forward(
        self,
        idx_q_BLHiDi: torch.Tensor,
        idx_k_BNDi: torch.Tensor,
        weights_BLHi: torch.Tensor,
        attention_masks: DeepSeekV41Metadata,
        *,
        candidates_BL1C: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """The score-and-select half, and the half a fused kernel replaces.

        Two passes, the shape the fused selector and V4's indices-only selection both imply:

        1. the relaxed score ``sum_h w * relu(q . k)`` over every entry the query can see,
           under ``no_grad`` -- selection is discrete, and this is the pass a kernel replaces;
        2. the hierarchy step: a Full Mode source builds the shared candidate pool, a Reindex
           Mode layer searches the pool it was handed.  Without ``candidate_topk_blocks`` both
           score everything visible;
        3. the top ``index_topk`` entries, numbered inside the query's own document;
        4. :func:`_student_logits` rescores those entries with gradient.

        ``candidates_BL1C`` carries the pool between layers as this reference's boolean mask
        ``[B, L, N]``, where a fused selector moves the kernels' block table.

        Returns ``(topk_indices_BLK, topk_scores_BLK, candidates_BL1C)``: document-local
        indices, ``-1`` for an unused slot, and ``-inf`` for that slot in the scores.
        """
        num_cmp = idx_k_BNDi.size(1)
        compressed = attention_masks.frame_for(self.compress_ratio)
        assert compressed is not None, (
            "A selector reads its compressed pool's frame, which only a layer that pools "
            f"owns, so compress_ratio must be positive (got {self.compress_ratio})."
        )
        cu_seqlens = attention_masks.q.cu_seqlens
        cu_seqlens_k = compressed.cu_seqlens
        visible_LN, newest_L1, newest_valid_L1, pool_start_L1 = _selection_mask(
            cu_seqlens,
            cu_seqlens_k,
            num_tokens=idx_q_BLHiDi.size(1),
            num_cmp=num_cmp,
            compress_ratio=self.compress_ratio,
        )

        with torch.no_grad():
            scores_BLHiN = torch.einsum("blhd,bnd->blhn", idx_q_BLHiDi, idx_k_BNDi)
            scores_BLN = (scores_BLHiN.relu() * weights_BLHi.unsqueeze(-1)).sum(dim=2)
            scores_BLN = scores_BLN.masked_fill(~visible_LN, -torch.inf)

            if self.mode is FULL and self.has_candidate_pool:
                candidates_BL1C = self.select_candidate_blocks(
                    scores_BLN,
                    newest_L1,
                    newest_valid_L1,
                    self.candidate_topk_blocks,
                    self.candidate_block_size,
                )
            elif self.mode is REINDEX and self.has_candidate_pool:
                assert candidates_BL1C is not None, (
                    "A Reindex Mode indexer with candidate_topk_blocks set searches the "
                    "pool the candidate source built, which no preceding layer produced."
                )
                scores_BLN = scores_BLN.masked_fill(~candidates_BL1C, -torch.inf)

            topk = min(self.index_topk, num_cmp)
            selected_BLK = scores_BLN.topk(topk, dim=-1, sorted=False).indices
            # Unreachable entries -- another document's, an incomplete group, or one a pool
            # excluded -- become -1, which the attention and the loss both skip; the mask
            # covers real entries only, so it is gathered at the picks.
            selected_valid_BLK = visible_LN.unsqueeze(0).gather(-1, selected_BLK)
            local_BLK = torch.where(selected_valid_BLK, selected_BLK - pool_start_L1, -1)
            # Emitted in the fused selector's order: position descending, so ``-1`` padding
            # lands at the tail.  The values alone are the emission; ``-1`` is the smallest
            # entry, so it takes the tail without a second index tensor.
            topk_indices_BLK = local_BLK.sort(dim=-1, descending=True).values

        # This gradient trains the indexer: into ``wq_b``, ``weights_proj`` and, through
        # ``idx_k``, the shared keys' owner.  Built on every path; who consumes it is the
        # teacher edge's decision.
        topk_scores_BLK = _student_logits(
            idx_q_BLHiDi,
            idx_k_BNDi,
            weights_BLHi,
            topk_indices_BLK,
            cu_seqlens,
            cu_seqlens_k,
        )
        return topk_indices_BLK, topk_scores_BLK, candidates_BL1C


class HierarchicalIndexer(Module):
    """The Hierarchical Sparse Indexer (report section 2.3.2) across CSA2's layer modes.

    V4.1 is a Causal Encoder-Decoder (CED): the bottom ``L/2`` layers are the causal
    encoder and the top ``L/2`` the decoder, and this indexer is used in the decoder
    only.  That is why the released config puts the candidate-pool source at the first
    decoder layer, ``candidate_source_layer = 20`` of 40.

    Each layer is statically assigned one mode (report section 2.3.1), and ``forward`` is
    the adapter over them:

    - ``FULL``: the layer owns the main KV, so it projects the index keys from the
      compressor latent and runs the indexer.  Its selector is the group's candidate
      source when it carries a pool capacity, and builds the shared pool.
    - ``REINDEX``: it reuses the main KV and index keys of a preceding Full Mode layer,
      computes its own index query and rescores them.  With a pool capacity its selector
      searches only the shared candidate pool, otherwise every visible entry.
    - ``REUSE``: no index query and no scores; it carries the latest Top-K indices and
      the shared keys forward.

    The candidate pool is the hierarchy: the source scores every causally visible entry
    once, scores each block by its best entry and keeps ``candidate_topk_blocks`` blocks
    of ``candidate_block_size`` positions -- 2048 x 8 = 16384 candidates against
    ``index_topk`` 512 in V4.1-Flash -- and the Reindex Mode layers search only those.
    The pool boundary comes from a top-k over blocks, so it receives no gradient: it is a
    training/inference consistency and per-query cost device, not a learned component.

    Which layer plays which part is the selector's business, not the indexer's: the pool
    capacity and the block size both live on ``Selector.Config``, and this adapter only
    carries the candidate tensor between layers.

    The indexer is trained by distillation alone, so its graph starts at its own
    parameters: the caller detaches the trunk inputs (``Attention.forward``).  The shared
    index keys are the one exception -- a Reindex Mode layer receives ``idx_k`` live, so
    that the consumers of a key keep training its owner.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mode: IndexerMode
        num_index_heads: int
        index_head_dim: int
        index_topk: int
        compress_ratio: int
        # Present on Full and Reindex Mode layers:
        rope: RoPE.Config | None = None
        wq_b: Linear.Config | None = None
        weights_proj: Linear.Config | None = None
        # Present on Full Mode layers, which project their own keys:
        selector: Selector.Config
        wk: Linear.Config | None = None
        k_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.mode = config.mode
        self.compress_ratio = config.compress_ratio
        self.num_index_heads = config.num_index_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.selector = config.selector.build()
        if self.mode is REUSE:
            return
        if config.rope is None or config.wq_b is None or config.weights_proj is None:
            raise ValueError("A Full or Reindex Mode indexer requires rope, wq_b and weights_proj configs.")
        self.rope = config.rope.build()
        self.wq_b = config.wq_b.build()
        self.weights_proj = config.weights_proj.build()
        if self.mode is FULL:
            if config.wk is None or config.k_norm is None:
                raise ValueError("A Full Mode indexer requires wk and k_norm configs to project its own index keys.")
            self.wk = config.wk.build()
            self.k_norm = config.k_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        attention_masks: DeepSeekV41Metadata,
        *,
        latent: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """The mode adapter.

        A Reuse Mode layer does no work at all: it hands its arguments back untouched.
        The other two modes project the index query and the per-head weights, and only
        Full Mode also projects the keys, from its own compressor latent.

        Args:
            x: Hidden states of shape ``[B, L, D]``.
            qr: Query LoRA latent of shape ``[B, L, Q]``.
            positions: Position ids of shape ``[B, L]``.
            attention_masks: The forward's varlen metadata; the indexer reads the packed
                frames, which say which pool entries a query's own document contributed.
            latent: The compressor's pre-RoPE latent; a Full Mode layer projects its keys
                from it.
            idx_k: The shared index keys, rescored by a Reindex Mode layer.
            topk_indices: The shared top-k, carried by a Reuse Mode layer.
            topk_scores: The shared student logits at those entries.  A Reuse Mode layer
                keeps them: they are what its own distillation loss compares its teacher
                against, and what the rest of its group inherits.
            candidates: The shared candidate pool, built by a Full Mode source and
                searched by the Reindex Mode layers after it.

        Returns:
            ``(idx_k, topk_indices, topk_scores, candidates)``.  A Reuse Mode layer passes
            its inputs through: it has no parameters of its own to train, but it must not
            drop the student logits its group depends on.
        """
        # Reuse Mode carries the shared selection forward: nothing to compute.  A layer
        # whose own ratio pools still consumes a selection, so being handed none means no
        # index source precedes it and the topology is wrong.
        if self.mode is REUSE:
            if self.compress_ratio > 0:
                assert topk_indices is not None, (
                    "A Reuse Mode indexer must be handed the shared top-k: no index "
                    f"source precedes this one (compress_ratio={self.compress_ratio})."
                )
            return idx_k, topk_indices, topk_scores, candidates

        # Only Full Mode projects keys, from its own compressor latent; Reindex Mode
        # rescores the keys a preceding Full Mode layer produced.
        if self.mode is FULL:
            assert latent is not None, "A Full Mode indexer projects its keys from the compressor latent."
            idx_k = self.k_norm(self.wk(latent))
            # One rank-2 head; RoPE rotates rank-3 [B, N, 1, H], with the entry's own
            # first token as its position.
            idx_k = self.rope(idx_k.unsqueeze(2), positions=positions[..., :: self.compress_ratio]).squeeze(2)
        else:
            assert idx_k is not None, (
                "A Reindex Mode indexer rescores the shared index keys, which no preceding Full Mode layer produced."
            )

        # The index query is rotated at the token's own position, which the metadata carries: the
        # positions beside the input describe the keys, and under CP the queries are a permuted
        # copy of those rows, so the two are not the same list of positions.
        idx_q = self.rope(
            self.wq_b(qr).unflatten(-1, (self.num_index_heads, self.index_head_dim)),
            positions=attention_masks.positions_q,
        )
        # Scaled by the index softmax scale and the head count, as in the reference: the
        # per-head scores are averaged rather than summed.
        weights = self.weights_proj(x) * (self.index_head_dim**-0.5 * self.num_index_heads**-0.5)

        topk_indices, topk_scores, candidates = self.selector(
            idx_q,
            idx_k,
            weights,
            attention_masks,
            candidates_BL1C=candidates,
        )
        return idx_k, topk_indices, topk_scores, candidates
