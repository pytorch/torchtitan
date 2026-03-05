# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight transfer planning for routed DTensor resharding.

Computes which chunks of sender (trainer) shards overlap with receiver
(generator) shards, enabling weight sync between different TP degrees.
Uses PyTorch DCP's resharding utilities for overlap detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.resharding import (
    _check_shard_metadata_pair_overlap,
    _shards_get_overlap_region_wrt_saved_tensor,
)


@dataclass
class TransferChunk:
    """Describes a single chunk transfer from sender to receiver."""

    param_name: str
    sender_rank: int
    receiver_rank: int
    sender_offset: tuple[int, ...]  # Offset within sender's local shard
    receiver_offset: tuple[int, ...]  # Offset within receiver's local shard
    lengths: tuple[int, ...]  # Size of chunk to transfer


@dataclass
class RankedChunk:
    """Wraps ChunkStorageMetadata with rank information."""

    rank: int
    chunk: ChunkStorageMetadata


def get_shard_metadata(
    global_shape: tuple[int, ...],
    num_ranks: int,
    shard_dim: int | None,
) -> list[RankedChunk]:
    """Compute RankedChunk for each rank given a shard dimension.

    Args:
        global_shape: Full tensor shape
        num_ranks: Number of ranks sharing this tensor
        shard_dim: Dimension to shard along, or None for replicated
    """
    shards = []

    if shard_dim is None:
        # Replicated: every rank has the full tensor
        for rank in range(num_ranks):
            chunk = ChunkStorageMetadata(
                offsets=torch.Size([0] * len(global_shape)),
                sizes=torch.Size(global_shape),
            )
            shards.append(RankedChunk(rank=rank, chunk=chunk))
        return shards

    dim_size = global_shape[shard_dim]
    chunk_size = (dim_size + num_ranks - 1) // num_ranks

    for rank in range(num_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, dim_size)
        actual_size = end - start

        offsets = [0] * len(global_shape)
        offsets[shard_dim] = start

        sizes = list(global_shape)
        sizes[shard_dim] = actual_size

        chunk = ChunkStorageMetadata(
            offsets=torch.Size(offsets),
            sizes=torch.Size(sizes),
        )
        shards.append(RankedChunk(rank=rank, chunk=chunk))

    return shards


def _compute_overlap(
    param_name: str,
    sender: RankedChunk,
    receiver: RankedChunk,
) -> TransferChunk | None:
    """Compute the overlap between a sender shard and receiver shard."""
    if not _check_shard_metadata_pair_overlap(sender.chunk, receiver.chunk):
        return None

    overlap_info = _shards_get_overlap_region_wrt_saved_tensor(
        sender.chunk, receiver.chunk
    )

    sender_offsets = tuple(info[1] for info in overlap_info)
    receiver_offsets = tuple(info[2] for info in overlap_info)
    lengths = tuple(info[3] for info in overlap_info)

    return TransferChunk(
        param_name=param_name,
        sender_rank=sender.rank,
        receiver_rank=receiver.rank,
        sender_offset=sender_offsets,
        receiver_offset=receiver_offsets,
        lengths=lengths,
    )


# ShardInfo: (global_shape, shard_dim or None)
ShardInfo = dict[str, tuple[tuple[int, ...], int | None]]


def compute_transfer_plans(
    sender_info: ShardInfo,
    sender_num_ranks: int,
    receiver_info: ShardInfo,
    receiver_num_ranks: int,
) -> tuple[dict[int, list[TransferChunk]], dict[int, list[TransferChunk]]]:
    """Compute sender and receiver transfer plans for all parameters.

    Args:
        sender_info: Per-param {name: (global_shape, shard_dim)} from trainer
        sender_num_ranks: Number of trainer ranks
        receiver_info: Per-param {name: (global_shape, shard_dim)} from generator
        receiver_num_ranks: Number of generator ranks

    Returns:
        (sender_plan, receiver_plan) where each is {rank: [TransferChunk, ...]}
    """
    sender_plan: dict[int, list[TransferChunk]] = {
        r: [] for r in range(sender_num_ranks)
    }
    receiver_plan: dict[int, list[TransferChunk]] = {
        r: [] for r in range(receiver_num_ranks)
    }

    for param_name, (global_shape, sender_shard_dim) in sender_info.items():
        receiver_shard_dim = receiver_info[param_name][1]

        sender_shards = get_shard_metadata(
            global_shape, sender_num_ranks, sender_shard_dim
        )
        receiver_shards = get_shard_metadata(
            global_shape, receiver_num_ranks, receiver_shard_dim
        )

        for recv_shard in receiver_shards:
            for send_shard in sender_shards:
                chunk = _compute_overlap(param_name, send_shard, recv_shard)
                if chunk is not None:
                    sender_plan[send_shard.rank].append(chunk)
                    receiver_plan[recv_shard.rank].append(chunk)

    return sender_plan, receiver_plan
