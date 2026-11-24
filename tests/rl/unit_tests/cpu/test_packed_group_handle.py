# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.rollout.packed_group_handle`: a rollout worker keeping a packed
group in a registered buffer (`RolloutGroupBuffers`) and a leader reading its rollouts back by
handle. Reads go over `LoopbackTransport` (`test_torch_transport_rdma.py`)."""

from __future__ import annotations

import pickle
from dataclasses import replace

import pytest

from torchtitan.rl.rollout.packed_group import (
    pack_rollout_group,
    PackedRolloutGroup,
    unpack_rollout_group,
)
from torchtitan.rl.rollout.packed_group_handle import (
    PackedRolloutGroupHandle,
    read_packed_rollout_group_handles,
    RolloutGroupBuffers,
)

from tests.rl.unit_tests.cpu.test_packed_group import (
    _assert_rollouts_equal,
    _group,
    _multiturn_group,
)
from tests.rl.unit_tests.cpu.test_torch_transport_rdma import loopback_buffers


def test_a_group_in_its_worker_reads_back_whole_from_its_handle_alone() -> None:
    """A leader gets only the handle (no summary); the regions hold each rollout's summary, so
    it reads back every field of every rollout, arrays and image files included."""
    owner, reader = loopback_buffers()
    group = _multiturn_group()

    _, handle = RolloutGroupBuffers(min_bytes=1, buffers=owner).pack(group)
    [read] = read_packed_rollout_group_handles([handle], reader)

    unpacked = unpack_rollout_group(read)
    _assert_rollouts_equal(unpacked.rollouts, group.rollouts)
    assert unpacked.rollouts[0].turns[0].completion_topk_logprobs.shape == (3, 4)
    regions = handle.regions()
    assert len(regions) == 3 + 1  # one per rollout, one per distinct image
    # Each region's bytes start 64-byte aligned, right after its leading tag.
    assert all((region.offset + 16) % 64 == 0 for region in regions)
    assert handle.num_bytes == sum(region.num_bytes for region in regions)
    # The handle names byte ranges only: no rollout, turn or array layout in it.
    handle_bytes = pickle.dumps(replace(handle, buffer=None))
    assert b"RolloutTurn" not in handle_bytes and len(handle_bytes) < 1024


def test_a_leader_reads_only_its_rollouts_and_their_images() -> None:
    owner, reader = loopback_buffers()
    group = _multiturn_group()
    _, handle = RolloutGroupBuffers(min_bytes=1, buffers=owner).pack(group)
    regions = handle.rollout_regions

    [read] = read_packed_rollout_group_handles([handle.select({1, 2})], reader)

    _assert_rollouts_equal(unpack_rollout_group(read).rollouts, group.rollouts[1:])
    # Rollouts 1 and 2 are adjacent: one read covering both, nothing of rollout 0 or the image.
    assert reader._transport.read_ranges == [
        (
            regions[1].offset,
            regions[2].offset + regions[2].num_bytes - regions[1].offset,
        )
    ]

    reader._transport.read_ranges.clear()
    [read] = read_packed_rollout_group_handles([handle.select({0})], reader)
    _assert_rollouts_equal(unpack_rollout_group(read).rollouts, group.rollouts[:1])
    # Rollout 0 and its image, not adjacent.
    assert len(reader._transport.read_ranges) == 2


def test_a_read_of_a_reused_buffer_raises_naming_both_tags() -> None:
    """A handle read after its buffer went to another group (a release-order bug) raises instead
    of returning the other group's bytes."""
    owner, reader = loopback_buffers()
    group_buffers = RolloutGroupBuffers(min_bytes=1, buffers=owner)
    _, old = group_buffers.pack(_group(group_size=2))
    group_buffers.release([old.group_id])
    _, new = group_buffers.pack(replace(_group(group_size=2), group_id=4))

    assert new.buffer.buffer_index == old.buffer.buffer_index
    with pytest.raises(RuntimeError, match=r"group 3 \(generation 1\).*\(4, 2\)"):
        read_packed_rollout_group_handles([old], reader)


def test_a_torn_region_raises() -> None:
    owner, reader = loopback_buffers()
    _, handle = RolloutGroupBuffers(min_bytes=1, buffers=owner).pack(_multiturn_group())
    region = handle.rollout_regions[1]
    tensor = owner.pool.buffer(handle.buffer.buffer_index).tensor
    tensor[region.offset + region.num_bytes - 1] += 1  # the trailing tag

    with pytest.raises(RuntimeError, match="reused the buffer"):
        read_packed_rollout_group_handles([handle], reader)


def test_rollout_group_buffers_keep_large_groups_and_reuse_released_buffers() -> None:
    owner, reader = loopback_buffers()
    group = _group(group_size=2)
    # Its regions alone: its buffer (with tags and alignment) is larger, and the 1-rollout
    # group's (one rollout region fewer) is smaller.
    group_buffers = RolloutGroupBuffers(
        min_bytes=pack_rollout_group(group)[1].num_bytes, buffers=owner
    )

    _, small = group_buffers.pack(replace(_group(group_size=1), group_id=5))
    summary, handle = group_buffers.pack(group)

    assert isinstance(small, PackedRolloutGroup)
    assert isinstance(handle, PackedRolloutGroupHandle) and handle.group_id == 3
    metrics = {metric.key: metric.value for metric in summary.group.metrics}
    assert metrics["rollout_worker/pool_registrations"].value == 1
    assert (
        metrics["rollout_worker/pool_bytes_in_use"].value == owner.pool.registered_bytes
    )
    [read] = read_packed_rollout_group_handles([handle], reader)
    _assert_rollouts_equal(unpack_rollout_group(read).rollouts, group.rollouts)

    group_buffers.release([3, 77])  # 77 is on another worker
    group_buffers.release([3])  # already released: nothing to do
    assert owner.pool.free_bytes == owner.pool.registered_bytes
    _, again = group_buffers.pack(_group(group_size=2))
    assert again.buffer.buffer_index == handle.buffer.buffer_index
    assert owner._transport.num_registrations == 1
    assert owner._transport.num_unregistrations == 0
