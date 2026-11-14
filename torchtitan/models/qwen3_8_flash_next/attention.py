# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen Sparse Attention with complete-block selection and indexer distillation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast, NamedTuple

import spmd_types as spmd

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed._functional_collectives import all_reduce, wait_tensor
from torch.utils.checkpoint import checkpoint

from torchtitan.distributed.spmd_types import spmd_mesh_group, spmd_mesh_size
from torchtitan.models.base.module import Module
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    AttentionMetadata,
    InnerAttention,
    local_head_split,
)
from torchtitan.models.common.attn_gym import gather_attn, lightning_indexer
from torchtitan.models.common.aux_loss import AuxLoss

from torchtitan.models.common.decoder_sharding import token_id_placement
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.qwen3_5.model import OffsetRMSNorm, Qwen35Attention
from torchtitan.models.qwen3_5.rope import MRoPE as Qwen35MRoPE

from torchtitan.tools.utils import has_cuda_capability

from .embedding import document_starts

# T = packed tokens, U = valid tokens, G = complete blocks, R = block size,
# N = document offset slots (document capacity + 1),
# B = selected blocks, J = selected tokens including the incomplete tail,
# H = query or KV heads, K = head dimension, V = value dimension,
# D = model dimension, Q = query chunk, C = projection channels.
# F = rotary features.


class SplitLinear(Linear):
    """Keep projection and gate parameters separate, with grouped output order."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        projection: Linear.Config
        gates: Linear.Config
        num_groups: int
        param_init: dict[str, Callable] | None = field(default_factory=dict)

        def __post_init__(self) -> None:
            if any(
                linear.bias or linear.num_linears != 1
                for linear in (self, self.projection, self.gates)
            ):
                raise ValueError("SplitLinear requires one bias-free projection")
            if self.num_groups < 1 or any(
                linear.out_features % self.num_groups
                for linear in (self.projection, self.gates)
            ):
                raise ValueError("SplitLinear projections must contain whole groups")
            if any(
                linear.in_features != self.in_features
                for linear in (self.projection, self.gates)
            ) or self.out_features != (
                self.projection.out_features + self.gates.out_features
            ):
                raise ValueError("SplitLinear dimensions must match its projections")

    def __init__(self, config: Config):
        super().__init__(config)
        del self.weight
        self.projection = config.projection.build()
        self.gates = config.gates.build()
        self.projection_channels_per_group = (
            config.projection.out_features // config.num_groups
        )
        # Fixed per-group widths keep the interleave valid on TP column shards.
        self.gate_channels_per_group = config.gates.out_features // config.num_groups

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        projection_THC = self.projection(input).unflatten(
            -1, (-1, self.projection_channels_per_group)
        )
        gates_THC = self.gates(input).unflatten(-1, (-1, self.gate_channels_per_group))
        remat.recompute_needs_tensor(projection_THC, gates_THC)
        return torch.cat((projection_THC, gates_THC), -1).flatten(-2)


class MRoPE(Qwen35MRoPE):
    """Reuse Qwen3.5's position selection with activation-dtype rotation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35MRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inverse:
            raise NotImplementedError("MRoPE does not support inverse rotation")
        rotary_dim = query.shape[-1]
        # NOTE: Preserve HF Flash-Next numerics by casting cos/sin to the
        # activation dtype before rotation. FP32 rotation followed by a final
        # cast changes BF16 rounding.
        cos_T1F = rope_cache[..., :rotary_dim].to(query.dtype)
        sin_T1F = rope_cache[..., rotary_dim:].to(query.dtype)

        def rotate(x_THF: torch.Tensor) -> torch.Tensor:
            first_THF, second_THF = x_THF.chunk(2, -1)
            return x_THF * cos_T1F + torch.cat((-second_THF, first_THF), -1) * sin_T1F

        query_THF = rotate(query)
        return query_THF if key is None else (query_THF, rotate(key))


def _selected_block_scores(query_QHK, key_GK, blocks_QB, *, scale):
    """Indexer logits of the selected blocks; -inf marks unused slots.

    One dense Q x G GEMM, then a gather. Gathering Q x B keys instead turns the
    backward into a scatter-add into key_GK, about 30x slower at T=4096.
    """
    scores_QHG = torch.einsum("qhk,gk->qhg", query_QHK, key_GK)
    scores_QG = scores_QHG.relu().sum(1) * scale
    return scores_QG.gather(-1, blocks_QB.clamp_min(0)).masked_fill(
        blocks_QB < 0, -torch.inf
    )


class QSAIndexMetadata(NamedTuple):
    """Document-local complete blocks and tails shared by all QSA layers.

    Shapes depend only on the token count and the document capacity, so the
    metadata can be refreshed in place for accelerator-graph replay. Complete
    blocks fill the first rows of ``block_indices_GR`` (G = T // R); the
    remaining rows are inactive capacity that point at a valid token and lie
    outside every document's candidate range, so they are never selected.
    Valid tokens are compacted in ``valid_indices_T``: its first U entries are
    their token indices and the rest are zero. Offsets have one slot per
    document of capacity; unused slots repeat the valid-token count.
    """

    valid_T: torch.Tensor  # noqa: N815
    document_T: torch.Tensor  # noqa: N815
    compact_T: torch.Tensor  # noqa: N815
    valid_indices_T: torch.Tensor  # noqa: N815
    cu_seqlens_N: torch.Tensor  # noqa: N815
    cu_seqlens_k_N: torch.Tensor  # noqa: N815
    block_indices_GR: torch.Tensor  # noqa: N815
    tail_indices_TR: torch.Tensor  # noqa: N815

    @classmethod
    def from_inputs(
        cls,
        document_positions_T,
        *,
        padding_mask_T,
        compress_ratio,
        max_num_documents=None,
    ):
        """Build the metadata; ``max_num_documents=None`` reserves T documents."""
        num_tokens = document_positions_T.numel()
        device = document_positions_T.device
        indices_T = torch.arange(num_tokens, device=device)
        starts_T = document_starts(document_positions_T)
        valid_T = (
            torch.ones_like(indices_T, dtype=torch.bool)
            if padding_mask_T is None
            else ~padding_mask_T
        )
        counts_T = valid_T.long().cumsum(0)
        start_counts_T = F.pad(counts_T[:-1], (1, 0))[starts_T]
        local_counts_T = counts_T - start_counts_T
        compact_T = (counts_T - 1).clamp_min(0)
        # Scatter targets past the kept range absorb rows that write nothing.
        valid_indices_T = torch.zeros(num_tokens + 1, dtype=torch.long, device=device)
        valid_indices_T.scatter_(
            0, torch.where(valid_T, compact_T, num_tokens), indices_T
        )
        valid_indices_T = valid_indices_T[:num_tokens]

        num_block_slots = num_tokens // compress_ratio
        block_end_T = valid_T & (local_counts_T.remainder(compress_ratio) == 0)
        block_end_counts_G = torch.zeros(
            num_block_slots + 1, dtype=torch.long, device=device
        )
        block_end_counts_G.scatter_(
            0,
            torch.where(block_end_T, block_end_T.long().cumsum(0) - 1, num_block_slots),
            counts_T,
        )
        block_indices_GR = valid_indices_T[
            (
                block_end_counts_G[:num_block_slots, None]
                - compress_ratio
                + torch.arange(compress_ratio, device=device)
            ).clamp_min(0)
        ]

        num_documents = num_tokens if max_num_documents is None else max_num_documents
        document_first_T = valid_T & (local_counts_T == 1)
        document_slot_T = document_first_T.long().cumsum(0) - 1
        if max_num_documents is not None:
            torch._assert_async(document_first_T.sum() <= max_num_documents)
        cu_seqlens_N = counts_T[-1:].repeat(num_documents + 2)
        cu_seqlens_N.scatter_(
            0,
            torch.where(document_first_T, document_slot_T, num_documents + 1),
            compact_T,
        )
        cu_seqlens_N = cu_seqlens_N[: num_documents + 1]
        cu_seqlens_k_N = F.pad(
            (cu_seqlens_N.diff() // compress_ratio).cumsum(0), (1, 0)
        ).int()
        document_T = document_slot_T.clamp(0, max(num_documents - 1, 0))

        tail_count_T = local_counts_T.remainder(compress_ratio)
        tail_offsets_TR = torch.arange(compress_ratio - 1, device=device).expand(
            num_tokens, -1
        )
        tail_slots_TR = counts_T[:, None] - tail_count_T[:, None] + tail_offsets_TR
        tail_indices_TR = valid_indices_T[tail_slots_TR.clamp_max(num_tokens - 1)]
        tail_indices_TR = tail_indices_TR.masked_fill(
            ~(valid_T[:, None] & (tail_offsets_TR < tail_count_T[:, None])), -1
        )
        return cls(
            valid_T,
            document_T,
            compact_T,
            valid_indices_T,
            cu_seqlens_N.int(),
            cu_seqlens_k_N,
            block_indices_GR,
            tail_indices_TR,
        )

    def annotate_spmd_types(self) -> None:
        types_T = token_id_placement()
        types_TR = spmd.SpmdType(
            {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
            spmd.PartitionSpec(("dp", "cp"), None),
        )
        for tensor in (
            self.valid_T,
            self.document_T,
            self.compact_T,
            self.valid_indices_T,
        ):
            spmd.assert_type(tensor, types_T)
        for tensor in (self.cu_seqlens_N, self.cu_seqlens_k_N):
            spmd.assert_type(tensor, {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R})
        spmd.assert_type(self.block_indices_GR, types_TR)
        spmd.assert_type(self.tail_indices_TR, types_TR)


class QSAIndexer(Module):
    """Score average-pooled complete blocks; always keep the incomplete tail."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        rotary_dim: int
        token_budget: int
        compress_ratio: int
        index_qk_proj: Linear.Config
        q_layernorm: OffsetRMSNorm.Config
        k_layernorm: OffsetRMSNorm.Config
        rope: MRoPE.Config
        # Dense fp32 Q x H x G scores per chunk: 16 MiB at T=4096, 128 MiB at 32K.
        chunk_size: int = 1024

        def __post_init__(self):
            if (
                min(self.num_heads, self.head_dim, self.compress_ratio, self.chunk_size)
                <= 0
            ):
                raise ValueError(
                    "Indexer head counts, dimensions and chunk sizes must be positive"
                )
            if (
                self.token_budget < self.compress_ratio
                or self.token_budget % self.compress_ratio
            ):
                raise ValueError(
                    "token_budget must be a positive multiple of compress_ratio"
                )
            if not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2:
                raise ValueError("rotary_dim must be positive, even and <= head_dim")

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.compress_ratio = config.compress_ratio
        self.block_topk = config.token_budget // config.compress_ratio
        self.chunk_size = config.chunk_size
        self.index_qk_proj = config.index_qk_proj.build()
        self.q_layernorm = config.q_layernorm.build()
        self.k_layernorm = config.k_layernorm.build()
        self.rope = config.rope.build()

    def _rotate(self, x_THK: torch.Tensor, positions_T: torch.Tensor) -> torch.Tensor:
        rotary_THR = x_THK[..., : self.rotary_dim]
        rotary_THR, _ = self.rope(rotary_THR, rotary_THR, positions_T)
        return torch.cat((rotary_THR, x_THK[..., self.rotary_dim :]), -1)

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        positions: torch.Tensor,
        document_positions_T: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
        metadata: QSAIndexMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x_TD.shape[0]
        qk_TC = self.index_qk_proj(x_TD)
        remat.recompute_needs_tensor(qk_TC)
        query_TC, raw_key_TK = qk_TC.split(
            (self.num_heads * self.head_dim, self.head_dim), -1
        )
        query_THK = self._rotate(
            self.q_layernorm(query_TC.unflatten(-1, (self.num_heads, self.head_dim))),
            positions,
        )
        if metadata is None:
            metadata = QSAIndexMetadata.from_inputs(
                document_positions_T,
                padding_mask_T=padding_mask_T,
                compress_ratio=self.compress_ratio,
            )
        block_indices_GR, tail_indices_TR = (
            metadata.block_indices_GR,
            metadata.tail_indices_TR,
        )
        num_blocks = block_indices_GR.shape[0]
        pooled_GK = raw_key_TK[block_indices_GR].float().mean(1).to(raw_key_TK.dtype)
        key_GK = self._rotate(
            self.k_layernorm(pooled_GK).unsqueeze(1), positions[block_indices_GR[:, 0]]
        ).squeeze(1)
        num_selected = min(self.block_topk, num_blocks)
        if num_selected:
            with torch.no_grad():
                blocks_TB = self._select_blocks(
                    query_THK, key_GK, metadata, num_selected=num_selected
                )
        else:
            blocks_TB = torch.full(
                (num_tokens, 0), -1, dtype=torch.long, device=x_TD.device
            )
        tokens_TBR = block_indices_GR[blocks_TB.clamp_min(0)].masked_fill(
            (blocks_TB < 0).unsqueeze(-1), -1
        )
        selected_indices_TJ = torch.cat(
            (tokens_TBR.flatten(-2), tail_indices_TR), -1
        ).int()
        selected_scores = []
        query_THK, key_GK = query_THK.float(), key_GK.float()
        for start in range(0, num_tokens, self.chunk_size):
            stop = start + self.chunk_size
            # Checkpoint per chunk: saving dense Q x G scores is quadratic storage.
            selected_scores.append(
                checkpoint(
                    _selected_block_scores,
                    query_THK[start:stop],
                    key_GK,
                    blocks_TB[start:stop],
                    scale=self.head_dim**-0.5,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            )
        return selected_indices_TJ, torch.cat(selected_scores)

    def _select_blocks(self, query_THK, key_GK, metadata, *, num_selected):
        """Top-k complete blocks per valid query with Attention Gym's lightning_indexer.

        lightning_indexer groups each document's tokens into compress_ratio-token
        candidates. Run it over the compacted valid tokens, so padding never forms
        or splits a block and the candidates are exactly ``QSAIndexMetadata``'s
        blocks; rows past the valid count return only -1. Scores rank as
        sum_h relu(q_h . k); the positive scale is irrelevant. ReLU makes exact
        ties common. Under deterministic algorithms the CuTe backend resolves
        them toward earlier blocks, independent of later candidates; otherwise
        the choice among cutoff ties is backend-defined.

        Returns:
            Selected global block ids per token in ascending order, -1 after
            the valid ones and for padding tokens.
        """
        compact_query_THK = query_THK[metadata.valid_indices_T]
        # The fused kernels need fp16/bf16 on SM90 or newer, as in DeepSeek-V4.
        fused = (
            compact_query_THK.is_cuda
            and compact_query_THK.dtype in (torch.float16, torch.bfloat16)
            and has_cuda_capability(9, 0)
        )
        compact_local_TB = lightning_indexer(
            compact_query_THK.unsqueeze(0),
            key_GK.unsqueeze(0),
            compact_query_THK.new_ones(1, *compact_query_THK.shape[:2]),
            num_selected,
            causal=True,
            compress_ratio=self.compress_ratio,
            cu_seqlens=metadata.cu_seqlens_N,
            cu_seqlens_k=metadata.cu_seqlens_k_N,
            impl="fused" if fused else "reference",
        ).squeeze(0)
        num_blocks = key_GK.shape[0]
        local_TB = compact_local_TB[metadata.compact_T]
        # Document-local -> global block ids; sort for a backend-independent order.
        blocks_TB = (
            torch.where(
                metadata.valid_T[:, None] & (local_TB >= 0),
                local_TB + metadata.cu_seqlens_k_N[metadata.document_T, None],
                num_blocks,
            )
            .sort(-1)
            .values
        )
        return blocks_TB.masked_fill(blocks_TB == num_blocks, -1).long()


class QSAIndexerLoss(AuxLoss):
    """KL loss on selected complete blocks from the report's sparse training stage."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        pass

    @staticmethod
    def block_kl(scores_QB: torch.Tensor, teacher_QB: torch.Tensor) -> torch.Tensor:
        valid_QB = scores_QB.isfinite()
        log_probs_QB = F.log_softmax(
            scores_QB.masked_fill(~valid_QB, torch.finfo(torch.float32).min), -1
        )
        teacher_QB = teacher_QB.detach() * valid_QB
        teacher_QB = teacher_QB / teacher_QB.sum(-1, keepdim=True).clamp_min(1e-30)
        return (torch.xlogy(teacher_QB, teacher_QB) - teacher_QB * log_probs_QB).sum()


class QSAInnerAttention(InnerAttention):
    """Attention over the indexer's selected tokens via Attention Gym's gather_attn.

    The selected blocks and the incomplete tail are sparse KV indices into the
    packed sequence; there is no sliding window. The indexer teacher is
    computed per query chunk of ``chunk_size`` to bound its Q x H x J logits.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        compress_ratio: int
        indexer_loss: QSAIndexerLoss.Config | None = None
        # 512 queries bound the gathered bf16 keys to ~1 GiB at J=2051.
        chunk_size: int = 512

        @property
        def is_full_attention(self) -> bool:
            return True

        def build_attention_metadata(
            self, positions, *, padding_mask=None, max_num_documents=None, **kwargs
        ) -> AttentionMetadata:
            # Decoder dispatches this metadata only to QSA by inner-attention type.
            return cast(
                AttentionMetadata,
                QSAIndexMetadata.from_inputs(
                    positions,
                    padding_mask_T=padding_mask,
                    compress_ratio=self.compress_ratio,
                    max_num_documents=max_num_documents,
                ),
            )

    def __init__(self, config: Config):
        super().__init__()
        self.compress_ratio = config.compress_ratio
        self.chunk_size = config.chunk_size
        self.indexer_loss = (
            config.indexer_loss.build() if config.indexer_loss is not None else None
        )

    def _teacher(self, query_THK, key_THK, selected_indices_TJ, num_blocks, *, scale):
        """Head-mean softmax over the selected tokens, max-pooled per complete block.

        Query head h reads KV head h // R. Per chunk, one gather of the selected
        keys feeds one batched GEMM per KV head over strided views, without
        replicating or transposing the gathered keys. On CUDA the GEMMs read bf16
        and accumulate and write fp32, exact up to summation order. They
        apply the scale and the mask: the fp32 minimum absorbs any finite
        logit, so masked slots equal it exactly. Under TP each rank holds a
        slice of the heads; their probability sums are all-reduced before max
        pooling, since pooling each shard first changes the target.
        """
        num_tokens, num_heads = query_THK.shape[:2]
        num_kv_heads, num_selected = key_THK.shape[1], selected_indices_TJ.shape[1]
        group_size = num_heads // num_kv_heads
        logits_dtype = torch.promote_types(query_THK.dtype, torch.float32)
        if not query_THK.is_cuda:
            # CPU has no mixed-dtype baddbmm; the reference path casts first.
            query_THK, key_THK = query_THK.to(logits_dtype), key_THK.to(logits_dtype)
        query_TGRK = query_THK.unflatten(1, (num_kv_heads, group_size))
        valid_TJ = selected_indices_TJ >= 0
        mask_T1J = (
            torch.zeros_like(valid_TJ, dtype=logits_dtype)
            .masked_fill_(~valid_TJ, torch.finfo(logits_dtype).min)
            .unsqueeze(1)
        )
        key_indices_TJ = selected_indices_TJ.clamp_min(0)
        num_global_heads = num_heads * spmd_mesh_size("tp")
        num_block_tokens = num_blocks * self.compress_ratio
        teacher_chunks = []
        for start in range(0, num_tokens, self.chunk_size):
            stop = min(start + self.chunk_size, num_tokens)
            keys_QJGK = key_THK[key_indices_TJ[start:stop]]
            logits_GQRJ = mask_T1J.new_empty(
                num_kv_heads, stop - start, group_size, num_selected
            )
            for g in range(num_kv_heads):
                gemm_args = (
                    mask_T1J[start:stop],
                    query_TGRK[start:stop, g],
                    keys_QJGK[:, :, g].transpose(1, 2),
                )
                if query_THK.is_cuda:
                    torch.baddbmm(
                        *gemm_args,
                        out_dtype=logits_dtype,
                        alpha=scale,
                        out=logits_GQRJ[g],
                    )
                else:
                    torch.baddbmm(*gemm_args, alpha=scale, out=logits_GQRJ[g])
            teacher_QJ = (
                logits_GQRJ.softmax(-1).sum((0, 2))
                / num_global_heads
                * valid_TJ[start:stop]
            )
            teacher_chunks.append(teacher_QJ[:, :num_block_tokens])
        teacher_TJ = torch.cat(teacher_chunks)
        tp_group = spmd_mesh_group("tp")
        if tp_group is not None:
            teacher_TJ = wait_tensor(all_reduce(teacher_TJ, "sum", tp_group))
        return teacher_TJ.unflatten(-1, (num_blocks, self.compress_ratio)).amax(-1)

    def forward(
        self,
        query_THK: torch.Tensor,
        key_THK: torch.Tensor,
        value_THV: torch.Tensor,
        *,
        selected_indices_TJ: torch.Tensor,
        scores_TB: torch.Tensor,
        scale: float,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        train_indexer = self.training and self.indexer_loss is not None
        if train_indexer and aux_loss_denominator is None:
            raise ValueError(
                "QSA training requires the step-global auxiliary-loss denominator"
            )
        with spmd.no_typecheck():
            # gather_attn takes [B, H, T, K]; query head h reads KV head
            # h // (H / H_kv). Padded query rows have no selections and return 0.
            output_1HTV = gather_attn(
                query_THK.transpose(0, 1).unsqueeze(0),
                None,
                key_THK.transpose(0, 1).unsqueeze(0),
                selected_indices_TJ.unsqueeze(0),
                sliding_window_size=0,
                scale=scale,
                impl="fused" if query_THK.device.type == "cuda" else "reference",
                sparse_v=value_THV.transpose(0, 1).unsqueeze(0),
            )
            output_THV = output_1HTV.squeeze(0).transpose(0, 1)
        # Kernel output is opaque to SPMD typechecking; it keeps the query layout.
        if spmd.is_type_checking():
            spmd.assert_type(
                output_THV,
                spmd.get_local_type(query_THK),
                spmd.get_partition_spec(query_THK),
            )
        if not train_indexer:
            return output_THV
        assert self.indexer_loss is not None
        num_blocks = scores_TB.shape[-1]
        if num_blocks:
            with torch.no_grad():
                teacher_TB = self._teacher(
                    query_THK, key_THK, selected_indices_TJ, num_blocks, scale=scale
                )
            raw_sum = self.indexer_loss.block_kl(scores_TB, teacher_TB)
        else:
            # Empty ranks must participate in the indexer's FSDP gradient
            # reduction; an empty sum is finite and keeps the graph edge.
            raw_sum = scores_TB.sum()
        tp_degree = spmd_mesh_size("tp")
        if tp_degree > 1:
            # Indexer parameters and scores are TP-replicated. The FSDP bridge
            # sums their local gradients, so contribute the global KL once.
            raw_sum = raw_sum.detach() + (raw_sum - raw_sum.detach()) / tp_degree
        return self.indexer_loss.inject(
            raw_sum, carrier=output_THV, denominator=aux_loss_denominator
        )


class QSAAttention(Qwen35Attention):
    """Qwen3.5's gated QK-normalized attention with the Flash-Next indexer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Attention.Config):
        inner_attention: QSAInnerAttention.Config  # pyrefly: ignore [bad-override]
        indexer: QSAIndexer.Config

        def __post_init__(self):
            Qwen35Attention.Config.__post_init__(self)
            # Decoder caches one metadata entry per inner-attention class.
            # The released block size is shared by every Flash-Next QSA layer.
            if (
                self.indexer.compress_ratio != 4
                or self.inner_attention.compress_ratio != 4
            ):
                raise ValueError("Flash-Next QSA requires compress_ratio=4")

        def flops_per_token(self, seq_len: int) -> int:
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            attended_tokens = min(
                seq_len, self.indexer.token_budget + self.indexer.compress_ratio - 1
            )
            return quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                seq_len=attended_tokens,
            ) + 6 * self.indexer.num_heads * self.indexer.head_dim * (
                seq_len // self.indexer.compress_ratio
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.indexer = config.indexer.build()
        if config.inner_attention.indexer_loss is None:
            self.indexer.requires_grad_(False)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        x_TD: torch.Tensor,
        attention_metadata: QSAIndexMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        document_positions_T: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions is None:
            raise ValueError("QSA requires explicit rotary positions")
        x_TD = maybe_gather_tp_input(self, x_TD)
        with torch.set_grad_enabled(
            torch.is_grad_enabled()
            and self.training
            and self.inner_attention.indexer_loss is not None
        ):
            selected_indices_TJ, scores_TB = self.indexer(
                # Only the explicit KL objective updates the indexer.
                x_TD.detach(),
                positions=positions,
                document_positions_T=document_positions_T,
                padding_mask_T=padding_mask_T,
                metadata=attention_metadata,
            )
        query_gate_THC = local_head_split(
            self.wq(x_TD), self.head_dim * 2, cp_shard_dim=0
        )
        query_THK, gate_THV = query_gate_THC.chunk(2, -1)
        key_THK = local_head_split(self.wk(x_TD), self.head_dim, cp_shard_dim=0)
        value_THV = local_head_split(self.wv(x_TD), self.head_dim, cp_shard_dim=0)
        remat.recompute_needs_tensor(query_THK, key_THK)
        query_THK, key_THK = self._partial_rope(
            self.q_norm(query_THK), self.k_norm(key_THK), positions
        )
        output_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            query_THK,
            key_THK,
            value_THV,
            selected_indices_TJ=selected_indices_TJ,
            scores_TB=scores_TB,
            scale=self.scaling,
            aux_loss_denominator=aux_loss_denominator,
        )
        remat.recompute_needs_tensor(output_THV, gate_THV)
        output_TD = (output_THV * gate_THV.sigmoid()).flatten(-2)
        return self.wo(output_TD)
