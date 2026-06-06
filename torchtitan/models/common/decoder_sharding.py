# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd

from torchtitan.models.common.attention import FusedQKVLinear, GQAttention, QKVLinear
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout
from torchtitan.protocols.types import MeshAxisName

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def dense_param_placement(*, tp: spmd.PerMeshAxisSpmdType) -> SpmdLayout:
    """Placement for dense-path params/buffers.

    DP/CP axes are replicated. TP placement is caller-specified.
    DTensor backend unfolds the DP axis to shard/replicate DTensor storage axes.
    """
    return SpmdLayout(
        {
            DP: spmd.R,
            CP: spmd.R,
            TP: tp,
        }
    )


def dense_activation_placement(
    *,
    tp: spmd.PerMeshAxisSpmdType,
    cp: spmd.PerMeshAxisSpmdType = spmd.S(1),
) -> SpmdLayout:
    """Placement for dense-path activations.

    DP is batch-sharded. CP defaults to seq-sharded S(1); override to R/I
    for K/V after all-gather. TP placement is caller-specified.
    """
    return SpmdLayout(
        {
            DP: spmd.S(0),
            CP: cp,
            TP: tp,
        }
    )


def dense_sequence_parallel_placement() -> SpmdLayout:
    """SP activations with CP-TP shard ordering on sequence dimension."""
    return SpmdLayout(
        {
            DP: spmd.S(0),
            CP: spmd.S(1),
            TP: spmd.S(1),
        },
        partition_spec=(DP, (CP, TP), None),
    )


def lm_head_sharding_config(
    *,
    loss_parallel: bool,
    enable_sp: bool,
    chunked_loss: bool = True,
) -> ShardingConfig:
    """
    Sharding for the decoder lm_head projection.
    ChunkedCELoss manually writes the S(1)->R allgather before chunking.
    Normal decoder forward needs lm_head boundary to perform it.
    TODO(pianpwk): maybe move to pre-lm-head norm w/ ChunkedCELoss refactor.
    """
    if enable_sp and not chunked_loss:
        in_src = dense_sequence_parallel_placement()
    else:
        in_src = dense_activation_placement(tp=spmd.R)
    in_dst = dense_activation_placement(tp=spmd.R)

    out_src = dense_activation_placement(tp=spmd.S(-1))
    out_dst = out_src if loss_parallel else dense_activation_placement(tp=spmd.I)

    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": in_src},
        in_dst_shardings={"input": in_dst},
        out_src_shardings=out_src,
        out_dst_shardings=out_dst,
    )


def colwise_config() -> ShardingConfig:
    """ColwiseParallel: weight S(0), output S(-1)."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1)),
    )


def rowwise_config(*, output_sp: bool = False) -> ShardingConfig:
    """
    RowwiseParallel: weight S(1), bias R (no-op if bias absent).
    Output redistributes to S(1) (reduce-scatter) if SP on, else I (all-reduce).
    """
    out_tp = spmd.S(1) if output_sp else spmd.I
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.R),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=out_tp),
    )


def norm_config(*, enable_sp: bool) -> ShardingConfig:
    """
    Norm sharding.
    Weight is unsharded@TP: R if SP (pending BWD AR handled by FSDP), else I.
    """
    state = {"weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)}
    activation = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    return ShardingConfig(
        state_shardings=state,
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
    )


def set_qkv_linear_sharding(qkv_linear_cfg) -> None:
    """Colwise-shard each Q/K/V projection of a ``BaseQKVLinear``.

    Handles both ``QKVLinear`` (separate ``wq`` + ``wkv``) and
    ``FusedQKVLinear`` (single ``wqkv``).
    """
    if isinstance(qkv_linear_cfg, FusedQKVLinear.Config):
        qkv_linear_cfg.wqkv.sharding_config = colwise_config()
    elif isinstance(qkv_linear_cfg, QKVLinear.Config):
        qkv_linear_cfg.wq.sharding_config = colwise_config()
        qkv_linear_cfg.wkv.sharding_config = colwise_config()
    else:
        raise TypeError(
            f"set_qkv_linear_sharding requires QKVLinear.Config or "
            f"FusedQKVLinear.Config, got {type(qkv_linear_cfg).__name__}"
        )


def set_gqa_attention_sharding(attention_cfg, *, enable_sp: bool) -> None:
    """Standard GQA attention (``qkv_linear``/``wo``) TP sharding.

    Shared by llama3, qwen3, and llama4 -- all three have a GQA block whose
    ``forward(x, ...)`` takes ``x`` (per-SP layout, gathered to Replicate
    internally) and uses the attention layer's local RoPE cache.

    Callers that have additional attention sub-state (e.g. ``qk_norm``,
    ``sinks``) set those after calling this helper.
    """
    assert isinstance(attention_cfg, GQAttention.Config), (
        f"set_gqa_attention_sharding requires GQAttention.Config, "
        f"got {type(attention_cfg).__name__}"
    )
    x_src = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    x_dst = dense_activation_placement(tp=spmd.R)
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": x_src,
        },
        in_dst_shardings={
            "x": x_dst,
        },
    )
    if attention_cfg.rope is not None:
        attention_cfg.rope.sharding_config = ShardingConfig(
            state_shardings={"cache": dense_param_placement(tp=spmd.R)},
        )
    set_qkv_linear_sharding(attention_cfg.qkv_linear)
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)


def set_gqa_inner_attention_local_map(
    inner_attention_cfg, *, return_lse: bool = False
) -> None:
    """Install a ``LocalMapConfig`` on an inner-attention config.

    q/k/v arrive as ``(bs, seq, heads, head_dim)`` DTensors with heads
    TP-sharded (``Shard(2)``), regardless of SP. ``local_map`` converts them
    to local tensors before the kernel runs, then wraps outputs back.

    Declares placements over the full dense SPMD axis set (DP/CP/TP) so
    the LocalMap composes under ``full_dtensor`` (where the surrounding
    mesh is multi-axis); under non-full_dtensor, the (tp,)-only mesh only
    consumes the ``TP`` placement and the rest are ignored.

    Under ``full_dtensor`` + CP, q stays seq-sharded on the CP axis
    (``Shard(1)``) while k/v are ``Replicate`` on CP -- DTensor all-gathers
    k/v at the local_map boundary so the kernel sees full-length keys
    (matching the BlockMask's kv dimension). Q's local grad is naturally
    seq-sharded; k/v's local grads accumulate as ``Partial`` on CP and
    DTensor reduces them on the way out.

    ``return_lse=True`` is for kernels that return ``(output, lse)`` (e.g.,
    GPT-OSS's flash attention with ``return_lse=True``); both outputs share
    the same heads-sharded placement.
    """
    q_placements = dense_activation_placement(tp=spmd.S(2))
    kv_src_placements = dense_activation_placement(tp=spmd.S(2))
    kv_placements = dense_activation_placement(tp=spmd.S(2), cp=spmd.R)
    kv_grad_placements = dense_activation_placement(tp=spmd.S(2), cp=spmd.P)
    out_src: SpmdLayout | tuple[SpmdLayout, ...] = (
        (q_placements, q_placements) if return_lse else q_placements
    )
    inner_attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "q": q_placements,
            "k": kv_src_placements,
            "v": kv_src_placements,
        },
        in_dst_shardings={
            "q": q_placements,
            "k": kv_placements,
            "v": kv_placements,
        },
        out_src_shardings=out_src,
        local_map=LocalMapConfig(
            in_grad_placements=(q_placements, kv_grad_placements, kv_grad_placements),
        ),
    )


def set_dense_ffn_sharding(
    feed_forward_cfg,
    *,
    attn_x_placement: spmd.PerMeshAxisSpmdType,
    enable_sp: bool,
) -> None:
    """Standard dense FFN (``w1``/``w2``/``w3``) TP sharding.

    Shared by llama3, qwen3, llama4, and deepseek_v3. ``attn_x_placement``
    should match the layout that the layer's attention block emits so the
    FFN's input wrap is a no-op redistribute when placements already agree.
    """
    in_src = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=attn_x_placement)
    )
    feed_forward_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": in_src},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
    )
    feed_forward_cfg.w1.sharding_config = colwise_config()
    feed_forward_cfg.w3.sharding_config = colwise_config()
    feed_forward_cfg.w2.sharding_config = rowwise_config(output_sp=enable_sp)


def set_decoder_sharding_config(
    config,
    *,
    loss_parallel: bool,
    enable_sp: bool,
    spmd_backend: str = "default",
    chunked_loss: bool = True,
) -> None:
    """Set sharding on root-level configs only: ``tok_embeddings``, ``norm``,
    and ``output``.

    Per-layer sharding (attention, feed_forward, per-layer norms) is the
    caller's responsibility — this helper does not walk ``config.layers``.

    ``enable_sp=True``  -> SequenceParallel: activations are ``Shard(1)`` between
    the embedding, norm, and output layers.
    ``enable_sp=False`` -> activations stay ``Replicate``; root norm is left
    unsharded (equivalent to the legacy ``NoParallel`` plan).
    """
    embed_input = dense_activation_placement(tp=spmd.R)
    embed_out = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    embed_src = dense_activation_placement(tp=spmd.P)
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": embed_input},
        in_dst_shardings={"input": embed_input},
        out_src_shardings=embed_src,
        out_dst_shardings=embed_out,
        # spmd_types backend relies on local SPMD + manual vocab-parallel embedding impl.
        local_map=LocalMapConfig(in_grad_placements=(None,))
        if spmd_backend == "spmd_types"
        else None,
    )
    config.norm.sharding_config = norm_config(enable_sp=enable_sp)

    config.lm_head.sharding_config = lm_head_sharding_config(
        loss_parallel=loss_parallel,
        enable_sp=enable_sp,
        chunked_loss=chunked_loss,
    )
