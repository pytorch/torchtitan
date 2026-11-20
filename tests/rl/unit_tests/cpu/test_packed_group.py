# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.rollout.packed_group`: packing a rollout group into one byte
buffer, selecting a leader's rollouts from it, and unpacking them."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, fields, replace

import numpy as np
from renderers import PlaceholderRange

from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.packed_group import pack_rollout_group, unpack_rollout_group
from torchtitan.rl.rollout.types import PromptImage, rollout_turn_lengths, RolloutTurnID
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

from tests.rl.unit_tests.cpu.test_finished_group import _group, _IMAGE_BYTES


@dataclass(kw_only=True, slots=True)
class _TurnWithTopkAndExperts(RolloutTurn):
    completion_topk_logprobs: np.ndarray
    routed_expert_ids: np.ndarray


def _multiturn_group(*, empty_completion: bool = True) -> RolloutGroup:
    """3 rollouts: one with an image and two turns, one with an empty completion (unless
    `empty_completion` is False: the batcher drops such a group), and turns that carry `[N, k]`
    top-k rows (a non-contiguous `[:, 1:]` view, as vLLM returns them) and a
    `uint8 [rows, 48, 8]` field, as router replay would."""
    rollouts = []
    for rollout_id in range(3):
        turns = []
        for turn_id in range(2 if rollout_id == 0 else 1):
            num_completion = 0 if rollout_id == 1 and empty_completion else 3 + turn_id
            topk = np.arange(num_completion * 5, dtype=np.float32).reshape(-1, 5)[:, 1:]
            with_image = rollout_id == 0 and turn_id == 0
            turns.append(
                _TurnWithTopkAndExperts(
                    rollout_id=RolloutTurnID(
                        group_id=9, rollout_id=rollout_id, turn_id=turn_id
                    ),
                    # Turn 1 continues turn 0's 4 prompt and 3 completion tokens.
                    prompt_prefix_len=7 * turn_id,
                    prompt_delta_token_ids=np.arange(4 + turn_id, dtype=np.int32),
                    prompt_delta_images=(
                        (
                            PromptImage(
                                hash_value="img",
                                source_path="a.png",
                                grid_thw=(1, 2, 2),
                                data=bytes(range(256)) * 3,
                            ),
                        )
                        if with_image
                        else ()
                    ),
                    prompt_delta_image_placeholders=(
                        (PlaceholderRange(offset=1, length=1),) if with_image else ()
                    ),
                    completion_token_ids=np.arange(num_completion, dtype=np.int32) + 10,
                    completion_logprobs=np.full(num_completion, -0.5, dtype=np.float32),
                    completion_topk_logprobs=topk,
                    routed_expert_ids=np.full(
                        (num_completion + 3, 48, 8), rollout_id, dtype=np.uint8
                    ),
                    min_policy_version=1,
                    max_policy_version=1,
                )
            )
        rollouts.append(
            Rollout(
                group_id=9,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                turns=turns,
                reward=float(rollout_id),
                advantage=float(rollout_id) - 1.0,
            )
        )
    return RolloutGroup(group_id=9, rollouts=rollouts)


def _assert_rollouts_equal(actual: list[Rollout], expected: list[Rollout]) -> None:
    """Every field of every rollout and turn equal, arrays and image file bytes included."""
    assert len(actual) == len(expected)
    for rollout, expected_rollout in zip(actual, expected, strict=True):
        assert replace(rollout, turns=[]) == replace(expected_rollout, turns=[])
        for turn, expected_turn in zip(
            rollout.turns, expected_rollout.turns, strict=True
        ):
            assert type(turn) is type(expected_turn)
            for turn_field in fields(turn):
                value = getattr(turn, turn_field.name)
                expected_value = getattr(expected_turn, turn_field.name)
                if isinstance(value, np.ndarray):
                    assert value.dtype == expected_value.dtype
                    np.testing.assert_array_equal(value, expected_value)
                else:
                    # By repr: `m.Metric` compares by identity, and unpacked turns hold copies.
                    assert repr(value) == repr(expected_value)
            # `PromptImage.data` is in neither its equality nor its repr.
            assert [image.data for image in turn.prompt_delta_images] == [
                image.data for image in expected_turn.prompt_delta_images
            ]


def test_a_packed_group_unpacks_every_field_of_every_rollout() -> None:
    """The buffer alone rebuilds the rollouts: each region holds its rollout's summary. The
    summary keeps each turn's class with every array cut to 0 rows."""
    group = _multiturn_group()

    summary, packed = pack_rollout_group(group)
    unpacked = unpack_rollout_group(pickle.loads(pickle.dumps(packed)))

    assert (unpacked.group_id, unpacked.metrics) == (9, [])
    _assert_rollouts_equal(unpacked.rollouts, group.rollouts)
    assert unpacked.rollouts[0].turns[0].completion_topk_logprobs.shape == (3, 4)
    # Writable: the buffer unpickles as a writable numpy array and the arrays are views of it.
    assert unpacked.rollouts[0].turns[0].completion_token_ids.flags.writeable
    assert summary.turn_lengths == tuple(
        rollout_turn_lengths(rollout) for rollout in group.rollouts
    )
    summary_turn = summary.group.rollouts[0].turns[0]
    assert isinstance(summary_turn, _TurnWithTopkAndExperts)
    assert summary_turn.completion_topk_logprobs.shape == (0, 4)
    assert summary_turn.routed_expert_ids.shape == (0, 48, 8)
    assert summary_turn.prompt_delta_images[0].data == b""
    assert packed.rollout_ids == (0, 1, 2)
    assert packed.rollout_image_hashes == (("img",), (), ())
    # Views into one buffer, each region 64-byte aligned in it.
    regions = [*packed.rollout_regions, *packed.image_regions.values()]
    buffer = regions[0].base
    assert all(region.base is buffer for region in regions)
    assert all(
        (region.ctypes.data - buffer.ctypes.data) % 64 == 0 for region in regions
    )


def test_a_packed_group_holds_each_image_file_once() -> None:
    """Each rollout's env reads the image file into its own bytes; the buffer keeps one copy."""
    group = _group()
    assert len(pickle.dumps(group)) > 8 * _IMAGE_BYTES

    summary, packed = pack_rollout_group(group)

    assert list(packed.image_regions) == ["h"]
    assert len(pickle.dumps((summary, packed))) < 2 * _IMAGE_BYTES


def test_select_keeps_only_the_named_rollouts_and_their_image_files() -> None:
    group = _group(group_size=3, image_per_rollout=True)
    _, packed = pack_rollout_group(group)

    selected = packed.select({0, 2})

    assert packed.select({0, 1, 2}) is packed
    assert (selected.group_id, selected.rollout_ids) == (3, (0, 2))
    assert list(selected.image_regions) == ["h0", "h2"]
    # No copy: the same region arrays. Pickling writes only theirs.
    assert selected.rollout_regions[1] is packed.rollout_regions[2]
    assert selected.image_regions["h2"] is packed.image_regions["h2"]
    assert selected.num_bytes < packed.num_bytes - _IMAGE_BYTES
    assert len(pickle.dumps(selected)) < len(pickle.dumps(packed)) - _IMAGE_BYTES
    _assert_rollouts_equal(
        unpack_rollout_group(selected).rollouts,
        [group.rollouts[0], group.rollouts[2]],
    )


def test_an_unpacked_group_builds_the_same_samples_as_the_group() -> None:
    """Specs decided from the summary (image files b"") build samples with the image files'
    bytes, taken from the unpacked rollouts."""
    group = _multiturn_group(empty_completion=False)
    builder = TrainingSampleBuilder(TrainingSampleBuilder.Config())
    summary, packed = pack_rollout_group(group)
    specs = builder.decide(
        rollout_group=summary.group,
        turn_lengths=summary.turn_lengths,
        num_mtp_layers=0,
    ).training_sample_specs

    expected = builder.materialize(training_sample_specs=specs, rollout_group=group)
    samples = builder.materialize(
        training_sample_specs=specs, rollout_group=unpack_rollout_group(packed)
    )
    image_data = {
        image.hash_value: image.data
        for rollout in group.rollouts
        for turn in rollout.turns
        for image in turn.prompt_delta_images
    }

    assert len(samples) == len(expected) > 0
    for sample, expected_sample in zip(samples, expected, strict=True):
        for sample_field in fields(sample):
            value = getattr(sample, sample_field.name)
            expected_value = getattr(expected_sample, sample_field.name)
            if isinstance(value, np.ndarray):
                assert value.dtype == expected_value.dtype
                np.testing.assert_array_equal(value, expected_value)
            else:
                assert value == expected_value
        # The image files' bytes (`PromptImage.data` is not in its equality).
        assert [image.data for image in sample.images] == [
            image_data[image.hash_value] for image in sample.images
        ]
    assert any(sample.images for sample in samples)
