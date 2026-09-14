# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.dist_gemm import (
    AsyncAllGatherLinear,
    AsyncAllGatherQKVLinear,
    AsyncLinearReduceScatter,
    validate_async_tp_preconditions,
)
from torchtitan.models.common.tensor_parallel import TensorParallelFeedForward
from torchtitan.protocols.sharding import ShardingConfig

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def dense_param_placement(*, tp: spmd.PerMeshAxisSpmdType) -> SpmdType:
    """Placement for dense-path params/buffers.

    DP/CP axes are spmd.R; the DTensor bridge unfolds DP into storage axes.
    TP placement is caller-specified.
    """
    return SpmdType(
        {
            DP: spmd.R,
            CP: spmd.R,
            TP: tp,
        }
    )


def dense_activation_placement(
    *,
    tp: spmd.PerMeshAxisSpmdType,
    cp: spmd.PerMeshAxisSpmdType,
) -> SpmdType:
    """Placement for dense-path activations.

    DP is token-sharded. CP and TP placements are caller-specified. Tensor
    dimensions not listed in the PartitionSpec are replicated.
    """
    cp_shards_tokens = isinstance(cp, spmd.Shard)
    tp_shards_features = isinstance(tp, spmd.Shard)
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V if cp_shards_tokens else cp,
            TP: spmd.V if tp_shards_features else tp,
        },
        partition_spec=spmd.PartitionSpec(
            (DP, CP) if cp_shards_tokens else DP,
            TP if tp_shards_features else None,
        ),
    )


def token_id_placement() -> SpmdType:
    """Placement for decoder token IDs with shape ``(tokens,)``."""
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.R,
        },
        partition_spec=spmd.PartitionSpec((DP, CP)),
    )


def token_id_sequence_parallel_placement() -> SpmdType:
    """Sequence-parallel token IDs with shape ``(tokens,)``.

    Same token-axis mesh as ``dense_sequence_parallel_placement()``, but the
    tensor is 1D so there is no trailing replicated feature dim.
    """
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=spmd.PartitionSpec((DP, CP, TP)),
    )


def attention_activation_placement(
    *, cp: spmd.PerMeshAxisSpmdType = spmd.S(0)
) -> SpmdType:
    """Placement for attention activations with shape ``(tokens, heads, dim)``."""
    if isinstance(cp, spmd.Shard):
        return SpmdType(
            {
                DP: spmd.V,
                CP: spmd.V,
                TP: spmd.V,
            },
            partition_spec=spmd.PartitionSpec((DP, CP), TP, None),
        )
    return SpmdType(
        {
            DP: spmd.S(0),
            CP: cp,
            TP: spmd.S(1),
        }
    )


def dense_sequence_parallel_placement() -> SpmdType:
    """Sequence-parallel ``(tokens, hidden)`` activation placement."""
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=spmd.PartitionSpec((DP, CP, TP), None),
    )


def decoder_input_sharding() -> dict[str, SpmdType]:
    """Default ``input_sharding`` for decoder-only models."""
    return {
        "input": token_id_placement(),
        "positions": token_id_placement(),
        "labels": SpmdType(
            {DP: spmd.V, CP: spmd.V, TP: spmd.I},
            partition_spec=spmd.PartitionSpec((DP, CP)),
        ),
    }


def colwise_config() -> ShardingConfig:
    """ColwiseParallel: weight S(0), output S(-1)."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0)),
    )


def rowwise_config(*, output_sp: bool = False) -> ShardingConfig:
    """
    RowwiseParallel: weight S(1), bias I (no-op if bias absent).
    Output redistributes to S(1) (reduce-scatter) if SP on, else I (all-reduce).
    """
    out_dst = (
        dense_sequence_parallel_placement()
        if output_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.I),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
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
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
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
    either the model forward or ``ChunkedLossWrapper`` applies ``lm_head``.
    """
    activation = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
        out_dst_shardings=dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
    )


def set_gqa_attention_sharding(attention_cfg, *, enable_sp: bool) -> None:
    """Standard GQA attention (``qkv_linear``/``wo``) TP sharding.

    Shared by llama3 and qwen3 -- both have a GQA block whose input uses the
    per-SP layout, is gathered to Replicate internally, and uses the attention
    layer's local RoPE cache.

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
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    common_gqa = attention_cfg._owner is GQAttention
    async_qkv = isinstance(attention_cfg.qkv_linear, AsyncAllGatherQKVLinear.Config)
    async_wo = isinstance(attention_cfg.wo, AsyncLinearReduceScatter.Config)
    if async_qkv != async_wo:
        raise ValueError(
            "Async tensor parallelism must configure both qkv and wo projections"
        )
    if async_qkv:
        validate_async_tp_preconditions(enable_sp=enable_sp)

    if common_gqa:
        # The projection leaves own the TP redistributions. The attention
        # wrapper only validates its external input and output layouts.
        attention_cfg.sharding_config = ShardingConfig(
            in_src_shardings={"x_TD": attn_x_layout},
            out_src_shardings=attn_x_layout,
        )
    else:
        # TODO: Muse Glimmer's GQAttention subclass shares the gathered input
        # between qkv and o_gate. Moving redistribution to both projection
        # leaves would duplicate the all-gather. Migrate this path once shared-
        # input communication has an explicit model boundary.
        attention_cfg.sharding_config = ShardingConfig(
            in_src_shardings={
                "x_TD": attn_x_layout,
            },
            in_dst_shardings={
                "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
            },
        )
    if attention_cfg.rope is not None:
        attention_cfg.rope.sharding_config = ShardingConfig(
            state_shardings={"cache": dense_param_placement(tp=spmd.R)},
        )

    if common_gqa:
        # The qkv projection now owns the input all-gather. Attaching the
        # redistribution here makes it part of the qkv module boundary.
        attention_cfg.qkv_linear.sharding_config = ShardingConfig(
            in_src_shardings={"x": attn_x_layout},
            in_dst_shardings=(
                None
                if async_qkv
                else {"x": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))}
            ),
        )
    attention_cfg.qkv_linear.wqkv.sharding_config = colwise_config()
    wo_sharding = rowwise_config(output_sp=enable_sp)
    attention_cfg.wo.sharding_config = (
        ShardingConfig(
            state_shardings=wo_sharding.state_shardings,
            out_src_shardings=wo_sharding.out_dst_shardings,
        )
        if async_wo
        else wo_sharding
    )


def set_gqa_inner_attention_local_spmd(inner_attention_cfg) -> None:
    """Localize TNH attention inputs without changing their placements.

    q/k use ``(T, H, K)`` and v uses ``(T, H, V)``. DP/CP shard T and TP
    shards H. CP collectives run inside the CP kernels; TP collectives still
    run at module boundaries.
    The local SPMD boundary converts annotated tensors to local tensors before
    the kernel runs, then wraps outputs back.

    TODO(fegin): drop the TP caveat once TP moves to the same mechanism.
    """
    placements = attention_activation_placement()
    inner_attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "q_THK": placements,
            "k_THK": placements,
            "v_THV": placements,
        },
        in_dst_shardings={
            "q_THK": placements,
            "k_THK": placements,
            "v_THV": placements,
        },
        out_src_shardings=placements,
        local_spmd=True,
    )


def set_dense_ffn_sharding(
    feed_forward_cfg,
    *,
    attn_x_layout: SpmdType,
    enable_sp: bool,
) -> None:
    """Standard dense FFN (physical ``w13``/``w2``) TP sharding.

    Shared by llama3, qwen3, and deepseek_v3. ``attn_x_layout`` should match
    the layout that the layer's attention block emits so the FFN's input wrap is
    a no-op redistribute when placements already agree.
    """
    tensor_parallel = isinstance(feed_forward_cfg, TensorParallelFeedForward.Config)
    async_w13 = isinstance(feed_forward_cfg.w13, AsyncAllGatherLinear.Config)
    async_w2 = isinstance(feed_forward_cfg.w2, AsyncLinearReduceScatter.Config)
    if async_w13 != async_w2:
        raise ValueError(
            "Async tensor parallelism must configure both w13 and w2 projections"
        )
    if async_w13:
        validate_async_tp_preconditions(enable_sp=enable_sp)
    if tensor_parallel:
        # The projection modules own the collectives inside their existing
        # remat regions. This wrapper only validates the external FFN contract.
        feed_forward_cfg.sharding_config = ShardingConfig(
            in_src_shardings={"x": attn_x_layout},
            out_src_shardings=attn_x_layout,
        )
        w13_sharding = colwise_config()
        feed_forward_cfg.w13.sharding_config = ShardingConfig(
            state_shardings=w13_sharding.state_shardings,
            in_src_shardings={"input": attn_x_layout},
            in_dst_shardings=(
                None
                if async_w13
                else {"input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))}
            ),
            out_src_shardings=w13_sharding.out_src_shardings,
        )
        w2_sharding = rowwise_config(output_sp=enable_sp)
        feed_forward_cfg.w2.sharding_config = (
            ShardingConfig(
                state_shardings=w2_sharding.state_shardings,
                out_src_shardings=w2_sharding.out_dst_shardings,
            )
            if async_w2
            else w2_sharding
        )
        return

    feed_forward_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": attn_x_layout},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
    )
    feed_forward_cfg.w13.sharding_config = colwise_config()
    feed_forward_cfg.w2.sharding_config = rowwise_config(output_sp=enable_sp)


def set_decoder_sharding_config(config, *, enable_sp: bool) -> None:
    """Set sharding on root-level configs only: ``tok_embeddings``, ``norm``,
    and ``output``.

    Per-layer sharding (attention, feed_forward, per-layer norms) is the
    caller's responsibility — this helper does not walk ``config.layers``.

    ``enable_sp=True``  -> SequenceParallel: activations are ``Shard(0)`` between
    the embedding, norm, and output layers.
    ``enable_sp=False`` -> activations stay ``Replicate``; root norm is left
    unsharded.
    """
    activation_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    embed_out_src = dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
    embed_input = token_id_placement()
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": embed_input},
        in_dst_shardings={"input": embed_input},
        out_src_shardings=embed_out_src,
        out_dst_shardings=activation_layout,
        local_spmd=True,
    )
    config.norm.sharding_config = pre_lm_head_norm_config(enable_sp=enable_sp)

    config.lm_head.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0)),
        out_dst_shardings=dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0)),
    )
