# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A finished training group packed into one byte buffer in its rollout worker, so the controller
keeps and forwards the group's bytes without unpickling them into Python objects.

    worker:      pack_rollout_group(group) -> (RolloutGroupSummary, PackedRolloutGroup)
    controller:  keeps the PackedRolloutGroup until the group's step, then sends each trainer DP
                 replica leader only the rollouts its column names (`PackedRolloutGroup.select`)
    leader:      unpack_rollout_group(packed) -> the rollouts, whole

The worker's buffer, from offset 0, every region 64-byte aligned:

    | rollout 0 | rollout 1 | ... | rollout G-1 | image a | image b |
      rollout region = [metadata length | pickled (rollout without arrays, array layout) |
                        padding | each turn's arrays, in turn order, each 64-byte aligned]
      image region = the image file's bytes
"""

from __future__ import annotations

import functools
import math
import pickle
import struct
from dataclasses import dataclass, fields, replace

import numpy as np

from torchtitan.rl.rollout.types import (
    Rollout,
    rollout_turn_lengths,
    RolloutGroup,
    RolloutGroupSummary,
)

# Every region, and every array in a rollout region, starts at a multiple of this many bytes.
_ALIGNMENT = 64
# What a rollout region starts with: the byte length of the pickled rollout and array layout
# after it.
_METADATA_LENGTH = struct.Struct("<q")

# Where one turn array sits in its rollout's region: (the `RolloutTurn` field, `np.dtype.str`,
# shape, bytes from the start of the region's arrays), e.g.
# ("completion_token_ids", "<i4", (221,), 256). A tuple, not a dataclass: a leader unpickles one
# per array, and a dataclass unpickles through a Python-level `__setstate__`.
_ArrayLayout = tuple[str, str, tuple[int, ...], int]


def _align(offset: int) -> int:
    return -(-offset // _ALIGNMENT) * _ALIGNMENT


@dataclass(frozen=True, slots=True)
class PackedRolloutGroup:
    """A rollout group as byte regions: one per rollout (its rollout and its arrays) and one per
    distinct image file. Besides the regions it holds only a few small tuples, which is what the
    controller keeps of a trainable group until its step.

    Each region is a uint8 numpy array: a view into the worker's buffer, and its own array once
    unpickled. Numpy arrays, not `bytes`, so they unpickle writable and the unpacked arrays
    (views into them) are too.

    Example:
        summary, packed = pack_rollout_group(group)  # 4 rollouts, one image
        packed.rollout_ids                            # (0, 1, 2, 3)
        packed.select({1, 3}).rollout_ids             # (1, 3): pickles only their regions
        unpack_rollout_group(packed.select({1, 3})).rollouts  # rollouts 1 and 3, with arrays
    """

    group_id: int

    rollout_ids: tuple[int, ...]
    """Each rollout region's `Rollout.rollout_id`."""

    rollout_regions: tuple[np.ndarray, ...]
    """uint8 [region bytes] per rollout, in `RolloutGroupSummary.group.rollouts` order."""

    rollout_image_hashes: tuple[tuple[str, ...], ...]
    """Per rollout region: the `PromptImage.hash_value` of each image its turns show."""

    image_regions: dict[str, np.ndarray]
    """`PromptImage.hash_value` -> uint8 [the image file's bytes]."""

    @property
    def num_bytes(self) -> int:
        return sum(region.nbytes for region in self.rollout_regions) + sum(
            region.nbytes for region in self.image_regions.values()
        )

    def select(self, rollout_ids: set[int]) -> PackedRolloutGroup:
        """Only the rollouts in `rollout_ids` (by `Rollout.rollout_id`) and the image files they
        show, with no copy: the same region arrays, fewer of them. What a DP replica leader
        gets: a group's rollouts can train on several replicas."""
        kept = [
            index
            for index, rollout_id in enumerate(self.rollout_ids)
            if rollout_id in rollout_ids
        ]
        if len(kept) == len(self.rollout_ids):
            return self
        image_hashes = {
            hash_value
            for index in kept
            for hash_value in self.rollout_image_hashes[index]
        }
        return PackedRolloutGroup(
            group_id=self.group_id,
            rollout_ids=tuple(self.rollout_ids[index] for index in kept),
            rollout_regions=tuple(self.rollout_regions[index] for index in kept),
            rollout_image_hashes=tuple(
                self.rollout_image_hashes[index] for index in kept
            ),
            image_regions={
                hash_value: region
                for hash_value, region in self.image_regions.items()
                if hash_value in image_hashes
            },
        )


@functools.cache
def _field_names(turn_type: type) -> tuple[str, ...]:
    return tuple(turn_field.name for turn_field in fields(turn_type))


def pack_rollout_group(
    group: RolloutGroup,
) -> tuple[RolloutGroupSummary, PackedRolloutGroup]:
    """Split `group` into its summary and its bytes: the bytes go into one buffer, with each
    rollout pickled into its region, so `unpack_rollout_group` rebuilds the rollouts.

    Every `np.ndarray`-valued field of a turn is an array, so per-token fields a feature adds
    (e.g. top-k logprobs) pack without code here, whatever their shape; a `None` field stays in
    the summary. Each region pickles its rollout with `None` for every array (an empty array
    costs more to pickle and unpickle, and unpacking replaces it anyway). The envs read each
    image file once per rollout, so a group of G rollouts holds G equal copies; the buffer keeps
    one per `hash_value`.

    Example:
        summary, packed = pack_rollout_group(group)
        summary.group.rollouts[0].turns[0].completion_token_ids  # array([], dtype=int32)
        summary.turn_lengths[0][0].num_completion_tokens          # 221
        unpack_rollout_group(packed).rollouts[0].turns[0].completion_token_ids  # the 221 ids
    """
    summary_rollouts = []
    rollout_image_hashes = []
    # Per rollout: its region's start and length in the buffer, its pickled metadata, where its
    # arrays start in its region, and its arrays.
    rollout_payloads = []
    image_data: dict[str, bytes] = {}
    position = 0
    for rollout in group.rollouts:
        summary_turns = []
        metadata_turns = []
        rollout_layout = []
        rollout_arrays = []
        num_array_bytes = 0
        for turn in rollout.turns:
            arrays = {
                name: value
                for name in _field_names(type(turn))
                if isinstance(value := getattr(turn, name), np.ndarray)
            }
            turn_layout: list[_ArrayLayout] = []
            for name, array in arrays.items():
                num_array_bytes = _align(num_array_bytes)
                turn_layout.append(
                    (name, array.dtype.str, array.shape, num_array_bytes)
                )
                num_array_bytes += array.nbytes
            for image in turn.prompt_delta_images:
                image_data.setdefault(image.hash_value, image.data)
            metadata_turn = replace(
                turn,
                **dict.fromkeys(arrays),
                prompt_delta_images=tuple(
                    replace(image, data=b"") for image in turn.prompt_delta_images
                ),
            )
            metadata_turns.append(metadata_turn)
            summary_turns.append(
                replace(
                    metadata_turn,
                    # Empty, with the same dtype and trailing shape (e.g. int32 [0, 32]).
                    **{
                        name: np.empty((0, *array.shape[1:]), dtype=array.dtype)
                        for name, array in arrays.items()
                    },
                )
            )
            rollout_layout.append(tuple(turn_layout))
            rollout_arrays.append(arrays)
        summary_rollouts.append(replace(rollout, turns=summary_turns))
        metadata = pickle.dumps(
            (replace(rollout, turns=metadata_turns), tuple(rollout_layout)),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        arrays_offset = _align(_METADATA_LENGTH.size + len(metadata))
        start = _align(position)
        position = start + arrays_offset + num_array_bytes
        rollout_image_hashes.append(
            tuple(
                dict.fromkeys(
                    image.hash_value
                    for turn in rollout.turns
                    for image in turn.prompt_delta_images
                )
            )
        )
        rollout_payloads.append(
            (
                start,
                position - start,
                metadata,
                arrays_offset,
                rollout_layout,
                rollout_arrays,
            )
        )
    image_spans = {}
    for hash_value, data in image_data.items():
        start = _align(position)
        position = start + len(data)
        image_spans[hash_value] = start

    buffer = np.zeros(position, dtype=np.uint8)
    rollout_regions = []
    for (
        start,
        num_bytes,
        metadata,
        arrays_offset,
        rollout_layout,
        rollout_arrays,
    ) in rollout_payloads:
        region = buffer[start : start + num_bytes]
        header = _METADATA_LENGTH.pack(len(metadata)) + metadata
        region[: len(header)] = np.frombuffer(header, dtype=np.uint8)
        for turn_layout, arrays in zip(rollout_layout, rollout_arrays, strict=True):
            for name, _, _, offset in turn_layout:
                array = arrays[name]
                array_start = arrays_offset + offset
                # Writes a non-contiguous array (e.g. a column slice) contiguous.
                np.copyto(
                    region[array_start : array_start + array.nbytes]
                    .view(array.dtype)
                    .reshape(array.shape),
                    array,
                )
        rollout_regions.append(region)
    image_regions = {}
    for hash_value, data in image_data.items():
        start = image_spans[hash_value]
        image_regions[hash_value] = buffer[start : start + len(data)]
        image_regions[hash_value][:] = np.frombuffer(data, dtype=np.uint8)
    summary = RolloutGroupSummary(
        group=replace(group, rollouts=summary_rollouts),
        turn_lengths=tuple(rollout_turn_lengths(rollout) for rollout in group.rollouts),
    )
    return summary, PackedRolloutGroup(
        group_id=group.group_id,
        rollout_ids=tuple(rollout.rollout_id for rollout in group.rollouts),
        rollout_regions=tuple(rollout_regions),
        rollout_image_hashes=tuple(rollout_image_hashes),
        image_regions=image_regions,
    )


def unpack_rollout_group(packed: PackedRolloutGroup) -> RolloutGroup:
    """The rollouts `packed` holds, whole (arrays and image files included), in a group without
    `metrics`. The arrays are views into `packed`'s regions."""
    image_data = {
        hash_value: region.tobytes()
        for hash_value, region in packed.image_regions.items()
    }
    rollouts = []
    for region in packed.rollout_regions:
        (num_metadata_bytes,) = _METADATA_LENGTH.unpack_from(region)
        metadata_end = _METADATA_LENGTH.size + num_metadata_bytes
        rollout, rollout_layout = pickle.loads(
            region[_METADATA_LENGTH.size : metadata_end]
        )
        arrays_start = _align(metadata_end)
        turn_arrays = [
            {
                name: region[
                    arrays_start
                    + offset : arrays_start
                    + offset
                    + _num_bytes(dtype, shape)
                ]
                .view(np.dtype(dtype))
                .reshape(shape)
                for name, dtype, shape, offset in turn_layout
            }
            for turn_layout in rollout_layout
        ]
        rollouts.append(_join_rollout(rollout, turn_arrays, image_data))
    return RolloutGroup(group_id=packed.group_id, rollouts=rollouts)


def _num_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    # math.prod, not np.prod: ~1 us instead of ~10 us, and a leader calls this per array.
    return math.prod(shape) * np.dtype(dtype).itemsize


def _join_rollout(
    rollout: Rollout,
    turn_arrays: list[dict[str, np.ndarray]],
    image_data: dict[str, bytes],
) -> Rollout:
    """A rollout without arrays with its turns' arrays and its images' file bytes put back."""
    return replace(
        rollout,
        turns=[
            replace(
                turn,
                **arrays,
                prompt_delta_images=tuple(
                    replace(image, data=image_data[image.hash_value])
                    for image in turn.prompt_delta_images
                ),
            )
            for turn, arrays in zip(rollout.turns, turn_arrays, strict=True)
        ],
    )
