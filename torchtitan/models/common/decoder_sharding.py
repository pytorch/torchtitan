# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName

from torchtitan.models.common.attention import FusedQKVLinear, GQAttention, QKVLinear
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def dense_param_placement(*, tp: spmd.PerMeshAxisSpmdType) -> SpmdLayout:
    """Placement for dense-path params/buffers.

    DP/CP axes are spmd.R; the DTensor bridge unfolds DP into storage axes.
    TP placement is caller-specified.
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
    """Sequence-parallel ``(batch, seq, hidden)`` activation placement."""
    return SpmdLayout(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=(DP, (CP, TP), None),
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
    out_dst = (
        dense_sequence_parallel_placement()
        if output_sp
        else dense_activation_placement(tp=spmd.I)
    )
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.R),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=out_dst,
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


def pre_lm_head_norm_config(*, enable_sp: bool) -> ShardingConfig:
    """Root decoder norm sharding before ``lm_head`` / chunked CE loss.

    Decoder blocks emit sequence-sharded hidden states when sequence
    parallelism is enabled. The root norm is the last clean module boundary to
    all-gather the TP sequence shard back to replicated hidden states before
    either the model forward or ``ChunkedCELoss`` applies ``lm_head``.
    """
    activation = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
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

    Shared by llama3 and qwen3 -- both have a GQA block whose
    ``forward(x_BLD, ...)`` takes ``x_BLD`` (per-SP layout, gathered to
    Replicate internally) and uses the attention layer's local RoPE cache.

    Callers that have additional attention sub-state (e.g. ``qk_norm``,
    ``sinks``) set those after calling this helper.
    """
    assert isinstance(attention_cfg, GQAttention.Config), (
        f"set_gqa_attention_sharding requires GQAttention.Config, "
        f"got {type(attention_cfg).__name__}"
    )
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "x_BLD": attn_x_layout,
        },
        in_dst_shardings={
            "x_BLD": dense_activation_placement(tp=spmd.R),
        },
    )
    if attention_cfg.rope is not None:
        attention_cfg.rope.sharding_config = ShardingConfig(
            state_shardings={"cache": dense_param_placement(tp=spmd.R)},
        )
    set_qkv_linear_sharding(attention_cfg.qkv_linear)
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)


def set_gqa_inner_attention_local_map(inner_attention_cfg) -> None:
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
    """
    q_placements: SpmdLayout = dense_activation_placement(tp=spmd.S(2))
    kv_src_placements: SpmdLayout = dense_activation_placement(tp=spmd.S(2))
    kv_dst_placements: SpmdLayout = dense_activation_placement(tp=spmd.S(2), cp=spmd.R)
    kv_grad_placements: SpmdLayout = dense_activation_placement(tp=spmd.S(2), cp=spmd.P)
    out_src: SpmdLayout = q_placements
    inner_attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "q_BLNH": q_placements,
            "k_BLNH": kv_src_placements,
            "v_BLNH": kv_src_placements,
        },
        in_dst_shardings={
            "q_BLNH": q_placements,
            "k_BLNH": kv_dst_placements,
            "v_BLNH": kv_dst_placements,
        },
        out_src_shardings=out_src,
        local_map=LocalMapConfig(
            in_grad_placements=(q_placements, kv_grad_placements, kv_grad_placements),
        ),
    )


def set_dense_ffn_sharding(
    feed_forward_cfg,
    *,
    attn_x_layout: SpmdLayout,
    enable_sp: bool,
) -> None:
    """Standard dense FFN (``w1``/``w2``/``w3``) TP sharding.

    Shared by llama3, qwen3, and deepseek_v3. ``attn_x_layout`` should match
    the layout that the layer's attention block emits so the FFN's input wrap is
    a no-op redistribute when placements already agree.
    """
    feed_forward_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": attn_x_layout},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
    )
    feed_forward_cfg.w1.sharding_config = colwise_config()
    feed_forward_cfg.w3.sharding_config = colwise_config()
    feed_forward_cfg.w2.sharding_config = rowwise_config(output_sp=enable_sp)


def set_decoder_sharding_config(config, *, enable_sp: bool) -> None:
    """Set sharding on root-level configs only: ``tok_embeddings``, ``norm``,
    and ``output``.

    Per-layer sharding (attention, feed_forward, per-layer norms) is the
    caller's responsibility — this helper does not walk ``config.layers``.

    ``enable_sp=True``  -> SequenceParallel: activations are ``Shard(1)`` between
    the embedding, norm, and output layers.
    ``enable_sp=False`` -> activations stay ``Replicate``; root norm is left
    unsharded (equivalent to the legacy ``NoParallel`` plan).
    """
    activation_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    embed_out_src = dense_activation_placement(tp=spmd.P)
    embed_input = dense_activation_placement(tp=spmd.R)
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": embed_input},
        in_dst_shardings={"input": embed_input},
        out_src_shardings=embed_out_src,
        out_dst_shardings=activation_layout,
        local_map=LocalMapConfig(in_grad_placements=None),
    )
    config.norm.sharding_config = pre_lm_head_norm_config(enable_sp=enable_sp)

    config.lm_head.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1)),
        out_dst_shardings=dense_activation_placement(tp=spmd.S(-1)),
    )
