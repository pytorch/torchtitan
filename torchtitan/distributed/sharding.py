# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Functions to fill ``ShardingSpec`` on model configs based on parallelism settings.

Called after ``expand_layer_configs()`` and ``update_from_config()``,
before ``build()``.  Each model has its own ``set_*_sharding_spec``
function that knows the model's structure.
"""

from torch.distributed.tensor import Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.protocols.sharding import LocalMapSpec, MeshDimName, ShardingSpec

TP = MeshDimName.TP


# ---------------------------------------------------------------------------
# Llama3
# ---------------------------------------------------------------------------


def set_llama3_sharding_spec(
    config,
    parallel_dims: ParallelDims,
    *,
    loss_parallel: bool,
) -> None:
    """Fill ``sharding_spec`` on all Llama3 sub-configs.

    No-op when TP is not enabled.
    """
    if not parallel_dims.tp_enabled:
        return

    _set_decoder_sharding_spec(config, loss_parallel)
    for layer_cfg in config.layers:
        _set_llama3_layer_sharding(layer_cfg)


# ---------------------------------------------------------------------------
# Shared decoder helpers (tok_embeddings, norm, output)
# ---------------------------------------------------------------------------


def _set_decoder_sharding_spec(config, loss_parallel: bool) -> None:
    """Set sharding on tok_embeddings, norm, output — shared by all decoders."""
    # tok_embeddings: RowwiseParallel — weight Shard(1), input Replicate, output Shard(1)
    config.tok_embeddings.sharding_spec = ShardingSpec(
        state_shardings={"weight": {TP: Shard(1)}},
        input_layouts={"input": {TP: Replicate()}},
        in_shardings={"input": {TP: Replicate()}},
        out_shardings={TP: Shard(1)},
    )
    # norm: SequenceParallel
    config.norm.sharding_spec = _sequence_parallel_spec()
    # output: ColwiseParallel — all-gather input before matmul
    config.output.sharding_spec = ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}},
        input_layouts={"input": {TP: Shard(1)}},
        in_shardings={"input": {TP: Replicate()}},
        out_shardings={TP: Shard(-1)} if loss_parallel else {TP: Replicate()},
    )


# ---------------------------------------------------------------------------
# Per-layer sharding
# ---------------------------------------------------------------------------


def _set_llama3_layer_sharding(layer_cfg) -> None:
    """Set sharding on one Llama3 transformer layer.

    Following PR 2149's full-DTensor approach:
    - All modules use use_local_output=False (DTensors flow everywhere)
    - ColwiseParallel outputs DTensor Shard(-1)
    - RowwiseParallel outputs DTensor Shard(1) (sequence parallel)
    - SequenceParallel norms keep DTensor
    - rope_cache is wrapped as DTensor Replicate via in_shardings
    """
    # Norms: SequenceParallel
    layer_cfg.attention_norm.sharding_spec = _sequence_parallel_spec()
    layer_cfg.ffn_norm.sharding_spec = _sequence_parallel_spec()

    # Attention: input x is Shard(1) from sequence-parallel norm,
    # needs all-gather to Replicate. rope_cache (freqs_cis) is a plain
    # tensor that must be wrapped as DTensor Replicate.
    layer_cfg.attention.sharding_spec = ShardingSpec(
        input_layouts={
            "x": {TP: Shard(1)},
            "rope_cache": {TP: Replicate()},
        },
        in_shardings={
            "x": {TP: Replicate()},
            "rope_cache": {TP: Replicate()},
        },
    )
    # wq/wkv: ColwiseParallel (DTensor output, no to_local)
    # wkv is the shared config for both wk and wv modules.
    for w in (
        layer_cfg.attention.wq,
        layer_cfg.attention.wkv,
    ):
        w.sharding_spec = _colwise_spec()
    # wo: RowwiseParallel with reduce-scatter to Shard(1)
    layer_cfg.attention.wo.sharding_spec = _rowwise_spec(out_shardings={TP: Shard(1)})

    # Inner attention: local_map for backends that don't support DTensor.
    # SDPA supports DTensor natively; FlexAttention and Varlen do not.
    inner_attn_cfg = layer_cfg.attention.inner_attention
    if isinstance(inner_attn_cfg, (FlexAttention.Config, VarlenAttention.Config)):
        qkv_placements = (Shard(1),)
        layer_cfg.attention.inner_attention_local_map = LocalMapSpec(
            in_placements=(qkv_placements, qkv_placements, qkv_placements),
            out_placements=(qkv_placements,),
            in_grad_placements=(
                qkv_placements,
                qkv_placements,
                qkv_placements,
            ),
        )

    # FFN: input x is Shard(1) from sequence-parallel norm,
    # needs all-gather to Replicate.
    assert layer_cfg.feed_forward is not None
    layer_cfg.feed_forward.sharding_spec = ShardingSpec(
        input_layouts={"x": {TP: Shard(1)}},
        in_shardings={"x": {TP: Replicate()}},
    )
    # w1, w3: ColwiseParallel (DTensor output)
    layer_cfg.feed_forward.w1.sharding_spec = _colwise_spec()
    layer_cfg.feed_forward.w3.sharding_spec = _colwise_spec()
    # w2: RowwiseParallel with reduce-scatter to Shard(1)
    layer_cfg.feed_forward.w2.sharding_spec = _rowwise_spec(
        out_shardings={TP: Shard(1)}
    )


# ---------------------------------------------------------------------------
# Reusable spec factories
# ---------------------------------------------------------------------------


def _colwise_spec() -> ShardingSpec:
    """ColwiseParallel: weight Shard(0), output Shard(-1)."""
    return ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}, "bias": {TP: Shard(0)}},
        out_shardings={TP: Shard(-1)},
    )


def _rowwise_spec(
    out_shardings: dict | None = None,
) -> ShardingSpec:
    """RowwiseParallel: weight Shard(1), output as specified."""
    if out_shardings is None:
        out_shardings = {TP: Replicate()}
    return ShardingSpec(
        state_shardings={"weight": {TP: Shard(1)}},
        out_shardings=out_shardings,
    )


def _sequence_parallel_spec() -> ShardingSpec:
    """SequenceParallel: weight Replicate, Shard(1) in, Shard(1) out."""
    return ShardingSpec(
        state_shardings={"weight": {TP: Replicate()}},
        input_layouts={"input": {TP: Shard(1)}},
        in_shardings={"input": {TP: Shard(1)}},
        out_shardings={TP: Shard(1)},
    )
