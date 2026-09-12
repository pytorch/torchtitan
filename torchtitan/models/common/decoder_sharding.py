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
    AsyncLinearReduceScatter,
    validate_async_tp_preconditions,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

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
    RowwiseParallel: weight S(1), bias R (no-op if bias absent).
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
            "bias": dense_param_placement(tp=spmd.R),
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
    if isinstance(
        attention_cfg.qkv_linear.wqkv, AsyncAllGatherLinear.Config
    ) or isinstance(attention_cfg.wo, AsyncLinearReduceScatter.Config):
        validate_async_tp_preconditions(enable_sp=enable_sp)

    if common_gqa:
        # GQAttention consumes these pairs explicitly inside its qkv and wo
        # remat regions. The generic wrapper only checks the rewritten source
        # and output contracts after the module is parallelized.
        attention_cfg.sharding_config = ShardingConfig(
            in_src_shardings={
                "x_TD": attn_x_layout,
            },
            in_dst_shardings={
                "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
            },
            out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
            out_dst_shardings=attn_x_layout,
        )
    else:
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
    attention_cfg.qkv_linear.wqkv.sharding_config = colwise_config()
    wo_config = rowwise_config(output_sp=enable_sp)
    if common_gqa:
        wo_config.out_dst_shardings = None
        if isinstance(attention_cfg.wo, AsyncLinearReduceScatter.Config):
            wo_config.out_src_shardings = attn_x_layout
    attention_cfg.wo.sharding_config = wo_config


def set_gqa_inner_attention_local_map(inner_attention_cfg) -> None:
    """Localize TNH attention inputs without changing their placements.

    q/k use ``(T, H, K)`` and v uses ``(T, H, V)``. DP/CP shard T and TP
    shards H. CP collectives run inside the CP kernels; TP collectives still
    run at module boundaries.
    ``local_map`` converts DTensors to local tensors before the kernel runs,
    then wraps outputs back.

    Placements include every SPMD axis. ``partial_dtensor`` uses only TP.

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
        local_map=LocalMapConfig(
            in_grad_placements=(placements, placements, placements),
        ),
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
    common_feed_forward = feed_forward_cfg._owner is FeedForward
    if isinstance(feed_forward_cfg.w1, AsyncAllGatherLinear.Config) or isinstance(
        feed_forward_cfg.w2, AsyncLinearReduceScatter.Config
    ):
        validate_async_tp_preconditions(enable_sp=enable_sp)
    if common_feed_forward:
        # FeedForward consumes these pairs explicitly inside its w13 and w2
        # remat regions. Model-specific subclasses retain their established
        # outer input boundary until their shared-input paths are audited.
        feed_forward_cfg.sharding_config = ShardingConfig(
            in_src_shardings={"x": attn_x_layout},
            in_dst_shardings={"x": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
            out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
            out_dst_shardings=attn_x_layout,
        )
    else:
        feed_forward_cfg.sharding_config = ShardingConfig(
            in_src_shardings={"x": attn_x_layout},
            in_dst_shardings={"x": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
        )
    feed_forward_cfg.w1.sharding_config = colwise_config()
    feed_forward_cfg.w3.sharding_config = colwise_config()
    w2_config = rowwise_config(output_sp=enable_sp)
    if common_feed_forward:
        w2_config.out_dst_shardings = None
        if isinstance(feed_forward_cfg.w2, AsyncLinearReduceScatter.Config):
            w2_config.out_src_shardings = attn_x_layout
    feed_forward_cfg.w2.sharding_config = w2_config


def set_decoder_sharding_config(config, *, enable_sp: bool) -> None:
    """Set sharding on root-level configs only: ``tok_embeddings``, ``norm``,
    and ``output``.

    Per-layer sharding (attention, feed_forward, per-layer norms) is the
    caller's responsibility — this helper does not walk ``config.layers``.

    ``enable_sp=True``  -> SequenceParallel: activations are ``Shard(0)`` between
    the embedding, norm, and output layers.
    ``enable_sp=False`` -> activations stay ``Replicate``; root norm is left
    unsharded (equivalent to the legacy ``NoParallel`` plan).
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
        local_map=LocalMapConfig(in_grad_placements=None),
    )
    config.norm.sharding_config = pre_lm_head_norm_config(enable_sp=enable_sp)

    config.lm_head.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0)),
        out_dst_shardings=dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0)),
    )
