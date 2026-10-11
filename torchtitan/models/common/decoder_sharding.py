# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig

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
    activation = residual_activation_placement(enable_sp=enable_sp)
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
    activation = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
        out_dst_shardings=dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
    )


def replicated_activation_placement() -> SpmdType:
    """``(tokens, hidden)`` activation gathered to Replicate across TP."""
    return dense_activation_placement(tp=spmd.R, cp=spmd.S(0))


def residual_activation_placement(*, enable_sp: bool) -> SpmdType:
    """Placement of the ``(tokens, hidden)`` residual stream between blocks.

    Sequence-parallel when ``enable_sp``; otherwise TP-invariant.
    """
    if enable_sp:
        return dense_sequence_parallel_placement()
    return dense_activation_placement(tp=spmd.I, cp=spmd.S(0))


def replicated_state_config(*names: str) -> ShardingConfig:
    """Replicate the named states across TP; no activation contract."""
    return ShardingConfig(
        state_shardings={name: dense_param_placement(tp=spmd.R) for name in names},
    )


def rope_cache_config() -> ShardingConfig:
    """Replicate a RoPE cache buffer across TP."""
    return replicated_state_config("cache")


def head_norm_config() -> ShardingConfig:
    """Norm over ``(tokens, heads, dim)`` activations sharded on heads.

    Used for QK norms; independent of sequence parallelism.
    """
    head_layout = attention_activation_placement()
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": head_layout},
        in_dst_shardings={"input": head_layout},
        out_src_shardings=head_layout,
        out_dst_shardings=head_layout,
    )


def gqa_attention_config(*, enable_sp: bool) -> ShardingConfig:
    """GQA attention boundary: ``x_TD`` in and out in the residual layout.

    The block gathers its input to Replicate internally and uses the
    attention layer's local RoPE cache.
    """
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x_TD": layout},
        out_src_shardings=layout,
    )


def inner_attention_local_spmd_config() -> ShardingConfig:
    """Localize TNH attention inputs without changing their placements.

    q/k use ``(T, H, K)`` and v uses ``(T, H, V)``. DP/CP shard T and TP
    shards H. CP collectives run inside the CP kernels; TP collectives still
    run at module boundaries.
    The local SPMD boundary converts annotated tensors to local tensors before
    the kernel runs, then wraps outputs back.

    TODO(fegin): drop the TP caveat once TP moves to the same mechanism.
    """
    placements = attention_activation_placement()
    return ShardingConfig(
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


def feed_forward_config(*, enable_sp: bool) -> ShardingConfig:
    """Dense FFN boundary: ``x`` in and out in the residual layout."""
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x": layout},
        out_src_shardings=layout,
    )


def token_embedding_config(*, enable_sp: bool) -> ShardingConfig:
    """Vocab-sharded token embedding whose output enters the residual layout.

    The local lookup produces a TP-Partial output that is reduced (and
    sequence-scattered under SP) at the module boundary.
    """
    embed_input = token_id_placement()
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": embed_input},
        in_dst_shardings={"input": embed_input},
        out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
        out_dst_shardings=residual_activation_placement(enable_sp=enable_sp),
        local_spmd=True,
    )


def replicated_input_colwise_config() -> ShardingConfig:
    """Column-parallel projection of an input already gathered to Replicate.

    Used where one module boundary gathers an input shared by several
    projections (e.g. MLA up-projections, gated attention projections).
    """
    return colwise_config(input_layout=replicated_activation_placement())


def lm_head_config() -> ShardingConfig:
    """Vocab-sharded ``lm_head`` over replicated hidden states."""
    replicated = replicated_activation_placement()
    vocab_sharded = dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0))
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": replicated},
        in_dst_shardings={"input": replicated},
        out_src_shardings=vocab_sharded,
        out_dst_shardings=vocab_sharded,
    )
