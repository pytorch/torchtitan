# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.attention import GQAttention
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


def token_id_placement(*, enable_sp: bool = False) -> SpmdType:
    """Placement for decoder token IDs with shape ``(tokens,)``.

    When sequence parallelism is enabled, TP also shards the token dimension.
    """
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V if enable_sp else spmd.R,
        },
        partition_spec=spmd.PartitionSpec((DP, CP, TP) if enable_sp else (DP, CP)),
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
        "padding_mask": token_id_placement(),
        "labels": SpmdType(
            {DP: spmd.V, CP: spmd.V, TP: spmd.I},
            partition_spec=spmd.PartitionSpec((DP, CP)),
        ),
    }


def colwise_config(*, input_layout: SpmdType) -> ShardingConfig:
    """Sharding contract for a column-parallel projection boundary."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": input_layout},
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0)),
    )


def stacked_colwise_config(*, input_layout: SpmdType) -> ShardingConfig:
    """Shard each ``[F, D]`` matrix in a ``[N, F, D]`` weight over ``F``.

    The input is ``[T, D]`` and the output is ``[T, N, F]``. DP and CP shard
    tokens while TP shards the per-matrix output features.
    """
    weight_NFD_layout = dense_param_placement(tp=spmd.S(1))
    bias_NF_layout = dense_param_placement(tp=spmd.S(1))
    output_TNF_layout = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        partition_spec=spmd.PartitionSpec((DP, CP), None, TP),
    )
    return ShardingConfig(
        state_shardings={
            "weight": weight_NFD_layout,
            "bias": bias_NF_layout,
        },
        in_src_shardings={"input": input_layout},
        out_src_shardings=output_TNF_layout,
        # Flattening [N, F, D] sharded on F produces a strided shard, which
        # global SPMD typechecking cannot represent. Keep the local projection
        # opaque while declaring its physical input and output layouts here.
        # ColumnParallelLinear still owns the input redistribution.
        local_spmd=True,
    )


def rowwise_config(
    *,
    output_layout: SpmdType,
) -> ShardingConfig:
    """Sharding contract for a row-parallel projection boundary."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.I),
        },
        in_src_shardings={
            "input": dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0))
        },
        out_src_shardings=output_layout,
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
    qkv = attention_cfg.qkv_linear.wqkv

    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x_TD": attn_x_layout},
        out_src_shardings=attn_x_layout,
    )
    qkv.sharding_config = colwise_config(input_layout=attn_x_layout)
    attention_cfg.wo.sharding_config = rowwise_config(output_layout=attn_x_layout)
    if attention_cfg.rope is not None:
        attention_cfg.rope.sharding_config = ShardingConfig(
            state_shardings={"cache": dense_param_placement(tp=spmd.R)},
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
    feed_forward_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": attn_x_layout},
        out_src_shardings=attn_x_layout,
    )
    feed_forward_cfg.w13.sharding_config = stacked_colwise_config(
        input_layout=attn_x_layout
    )
    feed_forward_cfg.w2.sharding_config = rowwise_config(output_layout=attn_x_layout)


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
