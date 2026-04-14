# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor import Replicate, Shard

from torchtitan.protocols.sharding import MeshDimName, ShardingSpec

TP = MeshDimName.TP


def colwise_spec() -> ShardingSpec:
    """ColwiseParallel: weight Shard(0), output Shard(-1)."""
    return ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}, "bias": {TP: Shard(0)}},
        out_shardings={TP: Shard(-1)},
    )


def rowwise_spec(
    out_shardings: dict | None = None,
) -> ShardingSpec:
    """RowwiseParallel: weight Shard(1), output as specified."""
    if out_shardings is None:
        out_shardings = {TP: Replicate()}
    return ShardingSpec(
        state_shardings={"weight": {TP: Shard(1)}},
        out_shardings=out_shardings,
    )


def sequence_parallel_spec() -> ShardingSpec:
    """SequenceParallel: weight Replicate, Shard(1) in, Shard(1) out."""
    return ShardingSpec(
        state_shardings={"weight": {TP: Replicate()}},
        input_layouts={"input": {TP: Shard(1)}},
        in_shardings={"input": {TP: Shard(1)}},
        out_shardings={TP: Shard(1)},
    )


def set_decoder_sharding_spec(config, loss_parallel: bool) -> None:
    """Set sharding on tok_embeddings, norm, output — shared by all decoders."""
    # tok_embeddings: RowwiseParallel — weight Shard(1), input Replicate, output Shard(1)
    config.tok_embeddings.sharding_spec = ShardingSpec(
        state_shardings={"weight": {TP: Shard(1)}},
        input_layouts={"input": {TP: Replicate()}},
        in_shardings={"input": {TP: Replicate()}},
        out_shardings={TP: Shard(1)},
    )
    # norm: SequenceParallel
    config.norm.sharding_spec = sequence_parallel_spec()
    # output: ColwiseParallel — all-gather input before matmul
    config.output.sharding_spec = ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}},
        input_layouts={"input": {TP: Shard(1)}},
        in_shardings={"input": {TP: Replicate()}},
        out_shardings={TP: Shard(-1)} if loss_parallel else {TP: Replicate()},
    )
