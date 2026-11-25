# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Fused override for the Qwen3.8-Flash-Next QSA indexer teacher.

The indexer's KL target is, per query, the head-mean softmax over its selected
tokens, max-pooled over each complete block's ``compress_ratio`` tokens.
``QSAInnerAttention._teacher`` gathers the selected keys into a ``[Q, J, H_kv, K]``
copy, runs skinny GEMMs to fp32 logits, then a separate softmax, head sum and
max-pool: at 125B shapes (16K tokens, J = 2051) it is memory-bound, 18.8 ms per
layer per microbatch.

``gather_attn`` already computes this softmax's normalizer for the attention
output, so each token's probability ``exp(S - lse_h)`` does not depend on which
other tokens are selected. A query selects only causal tokens of its own
document, at most ``max_context_length`` long, so this override computes the
head-summed probabilities densely per 64-query tile against the window before
it (one tensor-core GEMM per head and key tile, skipping key tiles before the
tile's first selected token), then gathers and max-pools the selected blocks:

```text
per 64-query tile, per 128-key tile of its window:
    S = scale * Q_h @ K_g^T               # per head h, KV head g = h // group
    sums += exp(S - lse_h)                # head sum, fp32
teacher[q, b] = max over block b's tokens of sums[q, token] / H   -> [T, B]
```

Under TP each rank holds a slice of the heads: it gathers the unpooled sums,
all-reduces them over TP, then pools, since pooling each shard first changes
the target.

At 125B shapes: 4.3 ms per layer per microbatch with one query per program
(selected keys gathered per query), 1.4-2.2 ms dense, depending on document
lengths.

TODO: this head-mean, block-max-pooled attention probability is generic to
sparse attention with an indexer teacher (DeepSeek V4's compressor teacher has
the same structure); it belongs in Attention Gym next to gather_attn.

Activate with::

    --override torchtitan_recipes.overrides.flash_next_qsa_teacher.fused_qsa_teacher
"""

from dataclasses import dataclass, replace

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
import triton
import triton.language as tl

from torch.distributed._functional_collectives import all_reduce, wait_tensor

from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_group, spmd_mesh_size
from torchtitan.models.common.attn_gym import AuxRequest, gather_attn
from torchtitan.models.qwen3_8_flash_next.attention import (
    QSAAttention,
    QSAInnerAttention,
)

__all__ = ["FusedTeacherQSAInnerAttention", "fused_qsa_teacher", "qsa_teacher"]

# Query and key tile sizes, from a sweep at 125B shapes on GB300 (8 warps).
_BLOCK_Q = 64
_BLOCK_N = 128


@triton.jit
def _teacher_kernel(
    query_ptr,
    key_ptr,
    lse_ptr,
    first_selected_ptr,
    out_ptr,
    num_tokens,
    scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_lh,
    stride_lt,
    stride_ot,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    CONTEXT: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_n = tl.program_id(1)
    q0 = pid_q * BLOCK_Q
    k0 = q0 - (CONTEXT - 1) + pid_n * BLOCK_N
    # The tile's selected tokens lie between its first selected token and its
    # last query; other key tiles are never read back.
    if k0 > q0 + BLOCK_Q - 1:
        return
    if k0 + BLOCK_N - 1 < tl.load(first_selected_ptr + pid_q):
        return
    offs_q = (q0 + tl.arange(0, BLOCK_Q)).to(tl.int64)
    offs_k = (k0 + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_q < num_tokens
    k_mask = (offs_k >= 0) & (offs_k < num_tokens)
    rows_k = tl.where(k_mask, offs_k, 0)
    sums = tl.zeros([BLOCK_Q, BLOCK_N], tl.float32)
    # Dynamic loops: unrolled heads would each hold a dot accumulator in
    # tensor memory.
    for g in range(NUM_HEADS // GROUP):
        key = tl.load(
            key_ptr + rows_k[:, None] * stride_kt + g * stride_kh + offs_d[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        for r in range(GROUP):
            head = g * GROUP + r
            query = tl.load(
                query_ptr
                + offs_q[:, None] * stride_qt
                + head * stride_qh
                + offs_d[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            logits = tl.dot(query, tl.trans(key)) * scale
            lse = tl.load(
                lse_ptr + head * stride_lh + offs_q * stride_lt, mask=q_mask, other=0.0
            )
            sums += tl.exp(logits - lse[:, None])
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + offs_q[:, None] * stride_ot + cols[None, :],
        sums,
        mask=q_mask[:, None],
    )


def qsa_teacher(
    query_THK: torch.Tensor,
    key_THK: torch.Tensor,
    selected_indices_TJ: torch.Tensor,
    lse_HT: torch.Tensor,
    *,
    num_blocks: int,
    compress_ratio: int,
    scale: float,
    max_context_length: int,
    num_global_heads: int,
) -> torch.Tensor:
    """Head-summed softmax over each query's block tokens / ``num_global_heads``.

    ``lse_HT`` is each head's log-sum-exp over all selected tokens (blocks and
    tail). A query selects only tokens of its own document, at most
    ``max_context_length`` long, so each 64-query tile computes the head-summed
    probabilities of the window before it and gathers its selected block
    tokens. Returns the per-token probabilities
    ``[T, num_blocks * compress_ratio]``, unpooled so TP can reduce them first.
    """
    num_tokens, num_heads, head_dim = query_THK.shape
    group = num_heads // key_THK.shape[1]
    assert head_dim == triton.next_power_of_2(head_dim)
    assert query_THK.stride(-1) == 1 and key_THK.stride(-1) == 1
    if num_tokens == 0:
        return query_THK.new_zeros(
            num_tokens, num_blocks * compress_ratio, dtype=torch.float32
        )
    # Block tokens only: the tail stays in the LSE but is not a block.
    block_tokens_TJ = selected_indices_TJ[:, : num_blocks * compress_ratio].long()
    num_tiles = triton.cdiv(num_tokens, _BLOCK_Q)
    first_selected = (
        F.pad(
            torch.where(block_tokens_TJ >= 0, block_tokens_TJ, num_tokens),
            (0, 0, 0, num_tiles * _BLOCK_Q - num_tokens),
            value=num_tokens,
        )
        .view(num_tiles, -1)
        .amin(-1)
    )
    # A document is no longer than the row either.
    context = min(max_context_length, num_tokens)
    num_key_tiles = triton.cdiv(context - 1 + _BLOCK_Q, _BLOCK_N)
    width = num_key_tiles * _BLOCK_N
    sums_TW = torch.empty(
        num_tokens, width, device=query_THK.device, dtype=torch.float32
    )
    _teacher_kernel[(num_tiles, num_key_tiles)](
        query_THK,
        key_THK,
        lse_HT,
        first_selected,
        sums_TW,
        num_tokens,
        scale,
        query_THK.stride(0),
        query_THK.stride(1),
        key_THK.stride(0),
        key_THK.stride(1),
        lse_HT.stride(0),
        lse_HT.stride(1),
        sums_TW.stride(0),
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        GROUP=group,
        CONTEXT=context,
        BLOCK_Q=_BLOCK_Q,
        BLOCK_N=_BLOCK_N,
        num_warps=8,
    )
    query_T = torch.arange(num_tokens, device=query_THK.device)
    window_start_T = query_T // _BLOCK_Q * _BLOCK_Q - (context - 1)
    column_TJ = block_tokens_TJ - window_start_T[:, None]
    valid_TJ = block_tokens_TJ >= 0
    # The kernel writes only the window's causal part: a selected token before
    # the window (a document longer than max_context_length) or after its query
    # would read an unwritten column.
    # TODO: torch._assert_async is private; switch to a public device-side
    # assert once PyTorch has one.
    torch._assert_async(
        (~valid_TJ | ((column_TJ >= 0) & (block_tokens_TJ <= query_T[:, None]))).all(),
        "QSA teacher: a selected token is non-causal or lies before its tile's "
        "window (a document is longer than max_context_length)",
    )
    return (
        sums_TW.gather(1, column_TJ.clamp(0, width - 1)).masked_fill(~valid_TJ, 0.0)
        / num_global_heads
    )


class FusedTeacherQSAInnerAttention(QSAInnerAttention):
    """``QSAInnerAttention`` whose indexer teacher reuses ``gather_attn``'s LSE.

    Same parameters (none) and attention output; only the KL target is computed
    differently.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QSAInnerAttention.Config):
        # Longest document the data can produce (the indexer's RoPE context);
        # sizes each query tile's window.
        max_context_length: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.max_context_length = config.max_context_length

    def _fused_teacher(
        self, query_THK, key_THK, selected_indices_TJ, lse_HT, num_blocks, *, scale
    ):
        probs_TJ = qsa_teacher(
            query_THK,
            key_THK,
            selected_indices_TJ,
            lse_HT,
            num_blocks=num_blocks,
            compress_ratio=self.compress_ratio,
            scale=scale,
            max_context_length=self.max_context_length,
            num_global_heads=query_THK.shape[1] * spmd_mesh_size("tp"),
        )
        tp_group = spmd_mesh_group("tp")
        if tp_group is not None:
            # Pooling each head shard first would change the target.
            probs_TJ = wait_tensor(all_reduce(probs_TJ, "sum", tp_group))
        return probs_TJ.unflatten(-1, (num_blocks, self.compress_ratio)).amax(-1)

    # Mirrors QSAInnerAttention.forward except for the LSE request and the
    # teacher call; keep the two in sync.
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
        if query_THK.device.type != "cuda":
            raise ValueError("FusedTeacherQSAInnerAttention requires CUDA")
        with spmd.no_typecheck():
            result = gather_attn(
                query_THK.transpose(0, 1).unsqueeze(0),
                None,
                key_THK.transpose(0, 1).unsqueeze(0),
                selected_indices_TJ.unsqueeze(0),
                sliding_window_size=0,
                scale=scale,
                impl="fused",
                sparse_v=value_THV.transpose(0, 1).unsqueeze(0),
                return_aux=AuxRequest(lse=True) if train_indexer else None,
            )
            output_1HTV, aux = result if train_indexer else (result, None)
            output_THV = output_1HTV.squeeze(0).transpose(0, 1)
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
            # Same retained region as the parent: the [T, B] target is kept, not
            # recomputed. lse is nondifferentiable fp32 from the fused kernel and,
            # like the kernel's output, opaque to SPMD typechecking.
            with spmd.no_typecheck():
                teacher_TB = remat.region(
                    self._fused_teacher, "indexer_teacher", recompute=False
                )(
                    query_THK.detach(),
                    key_THK.detach(),
                    selected_indices_TJ,
                    aux.lse.squeeze(0),
                    num_blocks,
                    scale=scale,
                )
            # The target is TP-reduced and keeps the scores' token layout.
            if spmd.is_type_checking():
                spmd.assert_type(
                    teacher_TB,
                    spmd.get_local_type(scores_TB),
                    spmd.get_partition_spec(scores_TB),
                )
            remat.recompute_needs_tensor(teacher_TB)
            raw_sum = self.indexer_loss.block_kl(scores_TB, teacher_TB)
        else:
            raw_sum = scores_TB.sum()
        tp_degree = spmd_mesh_size("tp")
        if tp_degree > 1:
            # As in the parent: TP ranks share the replicated indexer.
            raw_sum = raw_sum.detach() + (raw_sum - raw_sum.detach()) / tp_degree
        return self.indexer_loss.inject(
            raw_sum, carrier=output_THV, denominator=aux_loss_denominator
        )


@override(
    target=QSAAttention.Config,
    exact=True,
    description=(
        "Compute Flash-Next's QSA indexer teacher with one dense Triton pass from "
        "gather_attn's log-sum-exp."
    ),
)
def fused_qsa_teacher(cfg: QSAAttention.Config) -> QSAAttention.Config:
    # Targets the attention, not its inner attention, to read the indexer's
    # RoPE context.
    return replace(
        cfg,
        inner_attention=derive(
            cfg.inner_attention,
            FusedTeacherQSAInnerAttention.Config,
            max_context_length=cfg.indexer.rope.max_context_length,
        ),
    )
