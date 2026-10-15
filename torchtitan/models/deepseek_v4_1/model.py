# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4.1 backbones: the text stack and its multimodal extension.

Shape legend for this file:
    B = batch, L = sequence length, D = model dimension, hc = ``hc_mult`` residual
    branches.

:class:`DeepSeekV41Model` is the text-only backbone -- a plain
:class:`torchtitan.models.common.decoder.Decoder` with no MTP depths and no vision
tower.  :class:`DeepSeekV41MultimodalModel` subclasses it and adds the vision tower, the
image markers and the modality-routing bias; the only things that differ are the input
embedding construction and whether an ``image_mask`` reaches the block, so the text
stack carries no vision parameter, no vision branch and no vision import.

The block carries the residual stream as ``hc`` parallel branches and threads the
cross-layer attention state (compressed KV, index keys, selected entries, student
logits, candidate pool) explicitly: an attention source layer returns the tensors it
produced, every other layer returns what it was handed.  That keeps the reuse structure
visible in the forward signature instead of in mutable module state, at the cost of a
longer tuple.

Single-Pass mHC means each sublayer collapses its input with the mixing coefficients
predicted by the *previous* sublayer, so the block returns the coefficients its
successor needs.  The stack has no learned output head: the model collapses the
branches with the last block's coefficients and feeds the result straight to the output
norm.

Packed documents are described by :class:`DeepSeekV41Metadata`: one frame per addressed
tensor, whose packed boundaries are the only varlen metadata there is.  Both consumers read
those frames and nothing else -- the torch-native core and selector address the row with
them -- so a document never needs a second description.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import logging

import torch
from torch import nn
from torchtitan.models.common.decoder import Decoder, TransformerBlock

from .context_parallel import ExchangeMetadata, cp_plan, document_bounds
from .indexer import IndexerMode
from .metadata_extension import MetadataExtension
from .mhc import HcPost, HcPre

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE

    from .attention import Attention


def compression_alignment(compress_ratios: tuple[int, ...]) -> int:
    """The ``A`` a packed document length must be a multiple of, ``1`` when none.

    Every layer with ``compress_ratio > 0`` pools each group of ``R`` consecutive
    tokens into one main-KV entry, and the indexer addresses that entry axis as the row's
    boundaries over ``R``.  A document whose token count is not a multiple of ``R``
    therefore gets a partial trailing group that the *next* document's tokens complete:
    the pooled value mixes two documents, and the boundaries no longer say which document
    the group belongs to.  Keeping whole groups inside one document is what the alignment
    below is for.

    Taking the least common multiple of every ratio the stack pools with gives the
    alignment that keeps a whole group of each ratio inside one document; that is what
    the dataloaders pad each document to.
    """
    alignment = 1
    for ratio in compress_ratios:
        if ratio > 0:
            alignment = math.lcm(alignment, ratio)
    return alignment


@dataclass(frozen=True, kw_only=True, slots=True)
class KernelFrame:
    """One tensor's coordinate frame: the boundaries needed to address it.

    The tensors of a forward -- the query row and the two KV streams -- do **not** share a
    frame in general, so each carries its own and every consumer reads the frame of the tensor
    it is addressing rather than assuming one of them.

    ``cu_seqlens`` are the ragged document boundaries (``int32``, ``n_documents + 1``): the
    packed offsets Attention Gym's ``gather_attn`` takes as ``cu_seqlens``/``cu_seqlens_k``,
    and the form the kernels take as ``cu_seqlens_*``.  ``seqused`` is the per-document prefix
    this frame's consumer needs, ``residual`` the trailing partial group of a compressed axis
    (``None`` where the axis is not compressed, which the kernels reject a value for).

    **Context parallel.**  With one packed row per rank and no sharding, every frame is the
    row itself.  CP will move the query side's boundaries and the KV side's
    ``seqused``/``residual`` -- that is, this dataclass and nothing else -- which is why the
    frames are named after the tensor rather than after the field.
    """

    cu_seqlens: torch.Tensor
    seqused: torch.Tensor
    residual: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class DeepSeekV41Metadata:
    """Per-forward varlen metadata, built by :meth:`V41Model.get_attention_masks`.

    One geometry, read by both paths: one frame per addressed tensor -- the row's boundaries
    and each compressed stream's -- which the torch-native core and selector read exactly as
    the fused ports grid them.

    ``q`` and ``swa_k`` run at full resolution, so one frame each serves the whole forward.
    The compressed streams do not: every ratio>0 layer pools at *its own* ratio, so there is
    one frame per ratio the stack uses, looked up with :meth:`frame_for` rather than derived
    by each consumer.  At CP=1 every frame is the row itself; under context parallelism the
    query frame becomes this rank's chunks while the KV frames stay the gathered row, a
    difference the fused ports read through ``seqused``.  The torch-native core and selector
    are an unsharded path: a context-parallel run uses the fused ports.
    """

    q: KernelFrame
    swa_k: KernelFrame
    cmp_k: dict[int, KernelFrame]
    positions_q: torch.Tensor
    """The **query** frame's own positions.  The positions carried beside the input describe the
    keys -- under CP the queries are a permuted copy of the rows that carry them -- so every
    consumer that rotates a query reads them here, at every degree, without a fallback."""

    exchange: ExchangeMetadata | None = None
    """Under context parallelism the query side's movement: the query frame is this rank's
    chunks, and its ``positions_q`` carries the positions those chunks rotate at."""

    def frame_for(self, ratio: int) -> KernelFrame | None:
        """The frame of the compressed stream a layer at this ``ratio`` addresses.

        A ratio-1 layer still consumes a compressed stream, but it does not pool: its
        ``cmp_k`` is the row at full resolution, so its frame is ``q``'s and the kernels
        reject a ``residual`` for it.  Only ratios above 1 are pooled and stored.
        """
        if ratio <= 0:
            return None
        return self.cmp_k[ratio]


class DeepSeekV41TransformerBlock(TransformerBlock):
    """Transformer block with HC mixing around attention and the MoE feed-forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # Redeclared with the V4.1 types so config builders can reach the fields the
        # sharding and MTP helpers need.
        attention: Attention.Config  # pyrefly: ignore [bad-override]
        moe: MoE.Config  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config
        engram: object | None = None  # Optional Engram config; None for text-only.

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.moe_enabled = True
        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        self.moe = cfg.moe.build()
        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()
        self.engram = cfg.engram.build() if cfg.engram is not None else None

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: DeepSeekV41Metadata | None,
        positions: torch.Tensor | None = None,
        *,
        pre_mix: torch.Tensor,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        engram_token_history: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(x, pre_mix, cmp_k, idx_k, topk_indices, topk_scores, candidates)``.

        ``pre_mix`` is the attention-input coefficient the next block must consume, and
        the shared attention tensors are this block's contribution to the chain.
        ``image_mask`` is the block's only modality input and comes last, after the
        cross-layer attention state: the text stack's forward signature is that state
        and nothing else.  It is ``None`` whenever the caller has no modality, which is
        every text-only forward, and both consumers already read that as "no modality".
        """
        if self.engram is not None:
            x = self.engram(x, input_ids, positions, image_mask=image_mask, token_history=engram_token_history)
        residual = x
        x, attn_pre, post, comb = self.hc_attn_pre(x, pre_mix)
        x, cmp_k, idx_k, topk_indices, topk_scores, candidates = self.attention(
            self.attention_norm(x),
            positions,
            attention_masks,
            cmp_k=cmp_k,
            idx_k=idx_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            candidates=candidates,
        )
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, ffn_pre, post, comb = self.hc_ffn_pre(x, attn_pre)
        # MoE expects 2D [T, D]; flatten batch dim, then restore.
        moe_input = self.ffn_norm(x)
        B, L, D = moe_input.shape
        x = self.moe(moe_input.reshape(B * L, D), image_mask=image_mask).reshape(B, L, D)
        x = self.hc_post(x, residual, post, comb)
        return x, ffn_pre, cmp_k, idx_k, topk_indices, topk_scores, candidates


class DeepSeekV41Model(Decoder):
    """DeepSeek-V4.1 text backbone: CSA2 policy over the packed varlen contract.

    No MTP depths, no vision tower, no modality input: the stack consumes tokens,
    positions and the model-owned metadata, and nothing else.  The multimodal stack is
    :class:`DeepSeekV41MultimodalModel`, which only changes how the input embeddings are
    built and whether an ``image_mask`` reaches the blocks.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # Whether this stack can run under context parallelism: the text stack is the one
        # the CP layout is being built for, and the multimodal config overrides this to
        # ``False`` until the vision path has a CP story.
        #
        # The flag lives on the config because the gate that reads it runs on the config
        # tree, before any module exists.
        accepts_context_parallel: ClassVar[bool] = True

        n_layers: int
        hc_mult: int = 4
        compress_ratios: tuple[int, ...]
        kv_source_layers: tuple[int, ...] = ()
        index_source_layers: tuple[int, ...] = ()
        candidate_source_layer: int = 20
        candidate_topk_blocks: int = 2048
        candidate_block_size: int = 8
        metadata_extension: MetadataExtension.Config = field(default_factory=MetadataExtension.Config)
        """Backend post-process of the built masks, run once per microbatch in
        :meth:`build_attention_masks`.  The default is the identity; the fused
        recipes swap it via the ``metadata_extension`` override to precompute
        their kernel metadata, which is why the model dir fills its geometry
        (heads, head dim, window, index top-k, ratios) here rather than the
        override factory, which only sees this node."""

        def update_from_config(self, *, config, **kwargs):
            parallelism = config.parallelism
            if config.training.local_batch_size != 1:
                # V4.1 packs one row per rank: the document boundaries, the compressor's
                # group grid and the fused kernels' ragged layout are all defined on that
                # single row.  The model relies on it, so it is checked here rather than
                # left to each launcher.
                raise ValueError(
                    "DeepSeek V4.1 requires one packed row per rank "
                    f"(training.local_batch_size=1), got {config.training.local_batch_size}"
                )
            if parallelism.tensor_parallel_degree != 1:
                raise NotImplementedError("DeepSeek V4.1 currently supports TP=1 only")
            pp = parallelism.pipeline_parallel_degree
            if pp != 1:
                raise NotImplementedError(f"DeepSeek V4.1 does not support pipeline parallelism; got PP={pp}")
            # compile (aot_eager / inductor) is supported: the cross-layer
            # state is threaded explicitly and the sparse core is the
            # traceable gather_attn reference implementation.
            self._check_context_parallel(parallelism.context_parallel_degree)
            Decoder.Config.update_from_config(self, config=config, **kwargs)

            if len(self.compress_ratios) != self.n_layers:
                raise ValueError(
                    f"compress_ratios must match n_layers ({self.n_layers}), got {len(self.compress_ratios)}"
                )

            if hasattr(config, "training"):
                from torchtitan.models.common.rope import RoPE

                seq_len = config.training.seq_len
                for _, rope_cfg, _, _ in self.traverse(RoPE.Config):
                    setattr(rope_cfg, "max_seq_len", seq_len)  # noqa: B010

            engram_configs = [layer.engram for layer in self.layers if layer.engram is not None]
            ep_degree = max(1, parallelism.expert_parallel_degree)
            for engram_cfg in engram_configs:
                table_cfg = engram_cfg.table
                if (
                    table_cfg.require_token_id_map
                    and table_cfg.token_id_map_path is None
                    and table_cfg.tokenizer_path is None
                ):
                    if config.hf_assets_path is None:
                        raise ValueError(
                            "Engram tokenizer compression is required, but neither "
                            "table.token_id_map_path nor hf_assets_path is configured."
                        )
                    table_cfg.tokenizer_path = os.path.join(
                        config.hf_assets_path,
                        "tokenizer.json",
                    )
                if table_cfg.num_embeddings % ep_degree != 0:
                    raise ValueError(
                        f"Engram physical table size ({table_cfg.num_embeddings}) must "
                        f"be divisible by EP degree ({ep_degree}). Increase "
                        "EngramArgs.table_padding_multiple without changing it "
                        "between checkpoints."
                    )

            from .sharding import set_deepseek_v4_1_sharding_config

            set_deepseek_v4_1_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def _check_context_parallel(self, context_parallel_degree: int) -> None:
            """Reject a CP degree this stack cannot run.

            Context parallelism requires fused attention kernels that read
            ``seqused`` from sharded frames.  The reference path grids the
            whole row instead, so CP is rejected without fused ports.
            """
            if context_parallel_degree == 1:
                return
            if not type(self).accepts_context_parallel:
                raise NotImplementedError(
                    "DeepSeek V4.1 rejects context parallelism on the multimodal stack; "
                    f"got CP={context_parallel_degree}"
                )
            raise ValueError(
                "DeepSeek V4.1 runs context parallelism on the fused stack only: "
                f"got CP={context_parallel_degree} but no fused attention is configured. "
                f"Run with context_parallel_degree=1."
            )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            """Estimate V4.1 model FLOPs per token using TorchTitan's MFU convention."""
            from typing import cast

            from torchtitan.models.common.flops import (
                get_nparams_and_active_nparams,
                quadratic_attention_flops_per_token,
            )

            deepseek_v4_1_model = cast("DeepSeekV41Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(deepseek_v4_1_model)

            head_dim = self.layers[0].attention.head_dim
            num_flops_per_token = 6 * active_nparams
            for layer in self.layers:
                attention = layer.attention
                num_flops_per_token += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=attention.head_dim,
                    v_head_dim=attention.head_dim,
                    seq_len=seq_len,
                    sliding_window_size=attention.inner_attention.window_size,
                )
                if attention.compress_ratio > 0:
                    compressed_seq_len = seq_len // attention.compress_ratio
                    num_flops_per_token += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=attention.head_dim,
                        v_head_dim=attention.head_dim,
                        seq_len=min(attention.indexer.index_topk, compressed_seq_len),
                        sliding_window_size=0,
                    )
            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config
        self.hc_mult = cfg.hc_mult
        self.compress_ratios = tuple(cfg.compress_ratios)
        self.engram_history_size = max(
            (max(layer.engram.table.ngram_orders) - 1 for layer in cfg.layers if layer.engram is not None),
            default=0,
        ) if hasattr(cfg, 'layers') else 0
        # The per-microbatch metadata post-process (identity by default).  Built
        # here rather than at each call so a swapped extension keeps whatever
        # state it wants across batches; the model dir's geometry lives on the
        # config, which ``Module`` does not keep.
        self._metadata_extension = cfg.metadata_extension.build()
        # The precompute only pays off when a fused core reads the records: a
        # swapped extension over a reference (or quantized) attention stack is
        # dead work, so the pairing is flagged here, once, while the config
        # tree still exists -- a compile-time fact on the config classes, like
        # every other cross-node check the model makes.
        if type(cfg.metadata_extension) is not MetadataExtension.Config and not any(
            getattr(type(layer.attention.inner_attention), "reads_asc_metadata", False) for layer in cfg.layers
        ):
            logger.warning(
                "metadata_extension is overridden but no layer runs an attention core that "
                "reads the precomputed records: enable sparse_attn.asc alongside asc_metadata "
                "or drop the metadata override -- the per-microbatch precompute is dead work."
            )

    @staticmethod
    def _frame(axis_cu_seqlens: torch.Tensor, axis_lengths: torch.Tensor, ratio: int) -> KernelFrame:
        """One tensor's frame, from its own boundaries and its own per-document lengths.

        One frame per axis, and every axis is described the same way -- where its segments
        start, and how much of each segment it addresses -- so the ratio algebra lives here
        while the builders that call it differ only in where the two arrays come from.
        """
        if ratio == 1:
            # The uncompressed case: the frame is the tensor's own boundaries, and the kernels
            # reject a ``residual`` for an axis they do not compress.
            return KernelFrame(cu_seqlens=axis_cu_seqlens, seqused=axis_lengths)
        # A document need not be a whole number of groups: the compressed axis' boundaries are
        # the query's rounded down, and ``residual`` is what that rounding dropped, per document.
        return KernelFrame(
            cu_seqlens=(axis_cu_seqlens // ratio).to(torch.int32),
            seqused=(axis_lengths // ratio).to(torch.int32),
            residual=(axis_lengths % ratio).to(torch.int32),
        )

    def get_attention_masks(  # pyrefly: ignore [bad-override]
        self, positions: torch.Tensor
    ) -> DeepSeekV41Metadata:
        """Build the **unsharded** forward's metadata: every frame is the row itself.

        One grid, built once per forward and read by both paths: the torch-native core bounds
        its sliding window with the query frame's boundaries and addresses the pool with the
        compressed frame's, and the indexer derives its entry-axis reach from the same two --
        the frames the fused ports grid.  Building it must never depend on which of the two
        paths is enabled, which is what keeps a reference run and a fused run comparing the
        same axes.

        A sharded forward builds :meth:`_row_metadata` alone and cuts this rank's frames from
        it (:meth:`_sharded_attention_masks`), because the whole row is what the fused ports
        gather at their own boundary.
        """
        if positions is None:
            raise ValueError("DeepSeek V4.1 requires positions to build its attention metadata")
        return self._row_metadata(positions)

    def _row_metadata(self, positions: torch.Tensor) -> DeepSeekV41Metadata:
        """The whole row's metadata: one frame per addressed tensor, each of them the row.

        Called at every degree -- the sharded forward cuts this rank's frames from this grid
        rather than deriving it a second time -- so it stays free of anything only an unsharded
        forward reads.
        """
        cu_seqlens = document_bounds(positions)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        return DeepSeekV41Metadata(
            q=self._frame(cu_seqlens, lengths, 1),
            swa_k=self._frame(cu_seqlens, lengths, 1),
            cmp_k={
                ratio: self._frame(cu_seqlens, lengths, ratio)
                for ratio in sorted({r for r in self.compress_ratios if r > 0})
            },
            positions_q=positions,
        )

    def _sharded_attention_masks(
        self,
        metadata: DeepSeekV41Metadata,
        positions: torch.Tensor,
        *,
        cp_size: int,
        cp_rank: int,
    ) -> DeepSeekV41Metadata:
        """Upgrade the row's metadata to this rank's chunks: the sharded forward's metadata.

        The queries are chunk ``cp_rank`` of every document while the KV streams stay the
        **whole row** -- gathered declaratively at the fused core's boundary -- so the KV frames'
        ``cu_seqlens`` address that row and their ``seqused = K_d = (cp_rank + 1) · Q_d`` is how
        far these queries reach into each document.  That is what keeps the kernels' end-aligned
        window and causal limit anchored at this rank's chunk rather than at the row's end.

        ``metadata`` is :meth:`_row_metadata` of the same row -- the whole row's grid -- and it is
        where the boundaries come from: its query frame already carries them, so nothing is
        derived twice.  ``positions`` is the slab, whose own positions are not the query frame's.
        """
        bounds = metadata.q.cu_seqlens
        lengths = metadata.q.seqused
        # The launcher's contract, checked once per forward and per document: every document is a
        # whole multiple of ``cp_size * lcm(ratios > 1)``.  Both halves are load-bearing -- the
        # first makes every prefix a whole number of chunks, the second makes every compressed
        # prefix a whole multiple of the ratio -- and a violation is silent downstream: a ragged
        # chunk misplaces rows, and a ragged compressed prefix hands the kernels a partial group.
        # The value is CP-independent, so the same padding must be used at every degree compared.
        alignment = cp_size * math.lcm(*sorted({r for r in self.compress_ratios if r > 1}))
        if bool((lengths % alignment).any()):
            raise ValueError(
                f"a document length is not a multiple of cp_size * lcm(compress_ratios > 1) = "
                f"{alignment}: got {lengths.tolist()}. The dataloader's document alignment must "
                "be a multiple of that value, the same one at every context-parallel degree "
                "being compared (the padding is made of real tokens)."
            )
        chunk = (lengths // cp_size).to(torch.int32)
        # The slab is what the chunk grid has to reproduce: the row is cut into ``cp_size`` slabs
        # and the chunks are this rank's share of every document, so the two must size the same.
        if int(chunk.sum()) != positions.numel():
            raise ValueError(
                f"the sliced rows ({positions.numel()}) are not the chunks' total "
                f"({int(chunk.sum())}): the boundaries describe a different row than the one the "
                "inputs were sliced from."
            )
        reach = ((cp_rank + 1) * chunk).to(torch.int32)
        exchange = cp_plan(bounds, cp_size, cp_rank, positions.numel())
        return DeepSeekV41Metadata(
            q=self._frame((bounds // cp_size).to(torch.int32), chunk, 1),
            swa_k=self._frame(bounds, reach, 1),
            cmp_k={
                ratio: self._frame(bounds, reach, ratio) for ratio in sorted({r for r in self.compress_ratios if r > 0})
            },
            positions_q=exchange.positions_q.reshape(positions.shape),
            exchange=exchange,
        )

    def build_attention_masks(self, inputs, labels, extra_kwargs, *, cp_mesh=None, load_balancer_type=None):
        """Build the model-owned per-batch varlen metadata.

        The pinned trainer calls this hook for models that own their metadata instead of
        the Flex/Varlen ``get_attention_masks`` dispatch, so the model attaches its
        metadata here.
        """
        del load_balancer_type
        positions = extra_kwargs.get("positions")
        if cp_mesh is None:
            attention_masks = self.get_attention_masks(positions)
        else:
            # The whole row's grid is read here, before the slice, because it is the last point
            # at which the row exists -- and it is *all* that is built: a sharded forward has no
            # use for a description of the row itself.
            row = self._row_metadata(positions)
            cp_size, cp_rank = cp_mesh.size(), cp_mesh.get_local_rank()
            slab = positions.numel() // cp_size
            lo = cp_rank * slab
            if self.engram_history_size:
                # Inputs are still global here: retain only the left halo, not a
                # second copy of the full sequence. Hashing masks document resets.
                extra_kwargs["engram_token_history"] = inputs[:, max(0, lo - self.engram_history_size) : lo].clone()
            inputs = inputs[:, lo : lo + slab]
            labels = labels[:, lo : lo + slab]
            positions = positions[:, lo : lo + slab]
            extra_kwargs["positions"] = positions
            attention_masks = self._sharded_attention_masks(row, positions, cp_size=cp_size, cp_rank=cp_rank)
        # Every document must be a whole number of pooling groups, so the compressed axes
        # have no partial trailing group and ``cmp_residual_kv`` is zero throughout.  A
        # non-zero value here means the loader's document alignment and the stack's
        # ``compress_ratios`` disagree -- the kernels would then grid an axis the pooling
        # did not produce -- so it is checked rather than passed on.
        for ratio, frame in attention_masks.cmp_k.items():
            residual = frame.residual
            if residual is not None and bool(residual.any()):
                raise ValueError(
                    f"compress_ratio {ratio} has a partial trailing group: a document "
                    f"length is not a multiple of the pooling ratio ({residual.tolist()}). "
                    "The dataloader's document alignment must be a multiple of every "
                    "ratio in the stack's compress_ratios."
                )
        # The backend post-process runs once per microbatch, on whichever masks
        # the forward built (sharded or not): the default is the identity, and
        # the fused recipes swap it to precompute their kernel metadata at this
        # eager boundary instead of per layer inside the compiled region.
        if self._metadata_extension is not None:
            attention_masks = self._metadata_extension(attention_masks)
        extra_kwargs["attention_masks"] = attention_masks
        return inputs, labels, extra_kwargs

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context,
        parallelism,
        max_num_documents=None,
        max_context_length=None,
        **kwargs: Any,
    ) -> tuple:
        """Build V4.1 attention metadata and return the batch."""
        del kwargs
        positions = input_dict.get("positions")
        padding_mask = input_dict.get("padding_mask")
        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        if positions is not None:
            attention_masks = self.get_attention_masks(positions)
            input_dict["attention_masks"] = attention_masks
        return inputs, labels, input_dict

    def _embedded_inputs(
        self,
        tokens: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the input embedding stream of one forward.

        The text stack takes a caller-supplied stream or embeds the tokens itself.  The
        multimodal stack keeps this contract and derives the modality mask in its own
        override, because the mask is an input to the stack rather than part of the
        embedding stream.
        """
        tok_embeddings = self.tok_embeddings
        if input_embeds is not None:
            return input_embeds
        if tok_embeddings is None:
            return tokens
        return tok_embeddings(tokens)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: DeepSeekV41Metadata | None = None,
        *,
        input_embeds: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        engram_token_history: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        """V4.1 forward with Single-Pass mHC.

        The cross-layer attention state is threaded through the stack as ordinary local
        variables: an attention source layer returns the tensors it produced, and every
        other layer returns what it was handed.  ``image_mask`` is the stack's only
        modality input: it is ``None`` on the text stack, which never has one, and the
        multimodal stack hands it the mask it derives from ``token_types``.
        """
        hidden = self._embedded_inputs(tokens, input_embeds=input_embeds)
        input_ids = tokens.detach().long()
        # Ensure hidden has a batch dim: [B, L, D] or [L, D] -> [1, L, D]
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        pre_mix = HcPre.identity_pre_mix(hidden, self.hc_mult)
        cmp_k: torch.Tensor | None = None
        idx_k: torch.Tensor | None = None
        topk_indices: torch.Tensor | None = None
        topk_scores: torch.Tensor | None = None
        candidates: torch.Tensor | None = None
        for layer in self.layers.values():
            (
                hidden,
                pre_mix,
                cmp_k,
                idx_k,
                topk_indices,
                topk_scores,
                candidates,
            ) = layer(
                hidden,
                input_ids,
                attention_masks,
                positions,
                pre_mix=pre_mix,
                cmp_k=cmp_k,
                idx_k=idx_k,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
                candidates=candidates,
                image_mask=image_mask,
                engram_token_history=engram_token_history,
            )

        # The stack has no learned output head: it collapses with the last block's
        # attention-input coefficients.
        main_hidden = HcPre.collapse(hidden, pre_mix)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        # Flatten batch dim: loss and downstream expect 2D [T, D].
        if main_hidden.ndim == 3:
            main_hidden = main_hidden.reshape(-1, main_hidden.shape[-1])

        if self._skip_lm_head or self.lm_head is None:
            return main_hidden
        return self.lm_head(main_hidden)
