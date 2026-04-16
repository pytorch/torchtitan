# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.protocols.sharding import MeshDimName, ShardingSpec

TP = MeshDimName.TP


def colwise_spec() -> ShardingSpec:
    """ColwiseParallel: weight Shard(0), output Shard(-1)."""
    return ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}, "bias": {TP: Shard(0)}},
        out_shardings={TP: Shard(-1)},
    )


def rowwise_spec(*, output_sp: bool = False) -> ShardingSpec:
    """RowwiseParallel: weight Shard(1).

    ``output_sp=True``  → output ``Shard(1)`` (reduce-scatter into SP region).
    ``output_sp=False`` → output ``Replicate()`` (all-reduce).
    """
    output: Placement = Shard(1) if output_sp else Replicate()
    return ShardingSpec(
        state_shardings={"weight": {TP: Shard(1)}},
        out_shardings={TP: output},
    )


def sequence_parallel_spec() -> ShardingSpec:
    """SequenceParallel norm: weight Replicate, activations Shard(1)."""
    return ShardingSpec(
        state_shardings={"weight": {TP: Replicate()}},
        input_layouts={"input": {TP: Shard(1)}},
        in_shardings={"input": {TP: Shard(1)}},
        out_shardings={TP: Shard(1)},
    )


def replicate_norm_spec() -> ShardingSpec:
    """Plain-TP norm (no SP): weight Replicate, activations pass through.

    Needed so the norm's weight becomes a DTensor alongside DTensor
    activations; otherwise we'd mix plain Tensor and DTensor inside the op.
    """
    return ShardingSpec(state_shardings={"weight": {TP: Replicate()}})


def set_decoder_sharding_spec(config, *, loss_parallel: bool, enable_sp: bool) -> None:
    """Set sharding on tok_embeddings, norm, output — shared by all decoders.

    ``enable_sp=True``  → SequenceParallel: activations are ``Shard(1)`` between
    the embedding, norm, and output layers.
    ``enable_sp=False`` → activations stay ``Replicate``; root norm is left
    unsharded (equivalent to the legacy ``NoParallel`` plan).
    """
    act_sp: Placement = Shard(1) if enable_sp else Replicate()
    output_loss: Placement = Shard(-1) if loss_parallel else Replicate()
    config.tok_embeddings.sharding_spec = ShardingSpec(
        state_shardings={"weight": {TP: Shard(1)}},
        input_layouts={"input": {TP: Replicate()}},
        in_shardings={"input": {TP: Replicate()}},
        out_shardings={TP: act_sp},
    )
    config.norm.sharding_spec = (
        sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    )

    config.output.sharding_spec = ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}},
        input_layouts={"input": {TP: act_sp}},
        in_shardings={"input": {TP: Replicate()}},
        out_shardings={TP: output_loss},
    )
