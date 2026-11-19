# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The batcher plans from `TrainingSampleSpec`s, built from lengths; the samples
`materialize` builds from the arrays must agree with them."""

from __future__ import annotations

import dataclasses
import weakref

import numpy as np
import pytest
import torch
from renderers import PlaceholderRange

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.models.qwen3_5 import QWEN3_5_VISION_PROCESSOR_KWARGS
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import (
    Prompt,
    PromptImage,
    Rollout,
    rollout_turn_lengths,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
    split_prompt,
)
from torchtitan.rl.trainer.batcher import Batcher, MicrobatchPacker
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleBuilder,
    TrainingSampleSpecGroup,
)

from tests.rl.unit_tests.cpu.test_rollout_utils import rollout_to_training_samples

_NAN = float("nan")


def _image(hash_value: str) -> PromptImage:
    return PromptImage(
        hash_value=hash_value,
        source_path=f"{hash_value}.png",
        grid_thw=(1, 4, 4),  # 4 placeholder tokens
        data=hash_value.encode() * 3,
    )


def _rollout(
    rollout_id: int,
    turns: list[tuple[list[int], list[float]]],
    *,
    group_id: int = 0,
    images: dict[int, tuple[tuple[PromptImage, int], ...]] | None = None,
    reward: float | None = None,
) -> Rollout:
    """A scored rollout from each turn's full prompt and completion logprobs (completion token
    ids are 200 + position). `images[turn]` lists the turn's full-prompt (image, offset)s."""
    previous = Prompt(token_ids=[])
    rollout_turns = []
    for turn_id, (prompt_token_ids, logprobs) in enumerate(turns):
        turn_images = (images or {}).get(turn_id, ())
        prompt = Prompt(
            token_ids=prompt_token_ids,
            images=tuple(image for image, _ in turn_images),
            image_placeholders=tuple(
                PlaceholderRange(offset=offset, length=4) for _, offset in turn_images
            ),
        )
        prefix_len, delta = split_prompt(prompt, previous=previous)
        num_prefix_images = sum(
            placeholder.offset < prefix_len for placeholder in prompt.image_placeholders
        )
        completion = np.arange(200, 200 + len(logprobs), dtype=np.int32)
        rollout_turns.append(
            RolloutTurn(
                rollout_id=RolloutTurnID(
                    group_id=group_id, rollout_id=rollout_id, turn_id=turn_id
                ),
                prompt_prefix_len=prefix_len,
                prompt_delta_token_ids=delta,
                prompt_delta_images=prompt.images[num_prefix_images:],
                prompt_delta_image_placeholders=prompt.image_placeholders[
                    num_prefix_images:
                ],
                completion_token_ids=completion,
                completion_logprobs=np.array(logprobs, dtype=np.float32),
                min_policy_version=turn_id,
                max_policy_version=turn_id + 1,
            )
        )
        previous = Prompt(
            token_ids=prompt_token_ids + completion.tolist(),
            images=prompt.images,
            image_placeholders=prompt.image_placeholders,
        )
    return Rollout(
        group_id=group_id,
        rollout_id=rollout_id,
        status=RolloutStatus.COMPLETED,
        turns=rollout_turns,
        reward=float(rollout_id) if reward is None else reward,
        advantage=float(rollout_id) - 1.0,
    )


def _trainable_group() -> RolloutGroup:
    image_a, image_b = _image("a"), _image("b")
    return RolloutGroup(
        group_id=0,
        rollouts=[
            # A 1-token prompt; NaN logprobs at the start and the middle of the completion.
            _rollout(0, [([1], [_NAN, -0.1, _NAN, -0.3, -0.4])]),
            # Two turns in one sample (the second prompt extends the first), NaN in the middle.
            _rollout(
                1,
                [
                    ([1, 2, 3], [-0.1, -0.2]),
                    ([1, 2, 3, 200, 201, 9], [-0.1, _NAN, -0.3]),
                ],
            ),
            # A history edit: two samples.
            _rollout(2, [([1, 2], [-0.1]), ([9, 9], [-0.1, -0.2])]),
            # Images: b joins in turn 1 (carried with a), then swapped in place in turn 2.
            _rollout(
                3,
                [
                    ([1, 7, 7, 7, 7, 2], [-0.1]),
                    ([1, 7, 7, 7, 7, 2, 200, 3, 7, 7, 7, 7], [-0.2, -0.3]),
                    ([1, 7, 7, 7, 7, 2, 200, 3, 7, 7, 7, 7, 200, 201], [-0.4]),
                ],
                images={
                    0: ((image_a, 1),),
                    1: ((image_a, 1), (image_b, 8)),
                    2: ((image_b, 1), (image_b, 8)),
                },
            ),
            # Only NaN logprobs: no loss token, so its sample is dropped.
            _rollout(4, [([1, 2], [_NAN, _NAN])]),
            # Later turn without a completion: the rollout is dropped.
            _rollout(5, [([1], [-0.1]), ([1, 200, 4], [])]),
            # Three samples (history edits before turns 1 and 3). The middle one has only NaN
            # logprobs, so it is dropped: building walks its two turns without building them,
            # and the last sample reuses 3 of their tokens ([9, 2, 200]).
            _rollout(
                6,
                [
                    ([1, 2], [-0.1]),
                    ([9, 2], [_NAN, _NAN]),
                    ([9, 2, 200, 201, 3], [_NAN]),
                    ([9, 2, 200, 7], [-0.3, -0.4]),
                ],
            ),
        ],
        metrics=[m.Metric("rollout/x", m.Mean(1.0))],
    )


def _builder() -> TrainingSampleBuilder:
    return TrainingSampleBuilder.Config().build()


def _decide(
    builder: TrainingSampleBuilder, group: RolloutGroup, *, num_mtp_layers: int = 0
) -> TrainingSampleSpecGroup:
    return builder.decide(
        rollout_group=group,
        turn_lengths=tuple(map(rollout_turn_lengths, group.rollouts)),
        num_mtp_layers=num_mtp_layers,
    )


def _without_arrays(group: RolloutGroup) -> RolloutGroup:
    """`group` with every array field of every turn cut to 0 rows and every image file b""."""
    return dataclasses.replace(
        group,
        rollouts=[
            dataclasses.replace(
                rollout,
                turns=[
                    dataclasses.replace(
                        turn,
                        **{
                            turn_field.name: getattr(turn, turn_field.name)[:0]
                            for turn_field in dataclasses.fields(turn)
                            if isinstance(getattr(turn, turn_field.name), np.ndarray)
                        },
                        prompt_delta_images=tuple(
                            dataclasses.replace(image, data=b"")
                            for image in turn.prompt_delta_images
                        ),
                    )
                    for turn in rollout.turns
                ],
            )
            for rollout in group.rollouts
        ],
    )


def _assert_samples_equal(actual: TrainingSample, expected: TrainingSample) -> None:
    for sample_field in dataclasses.fields(expected):
        actual_value = getattr(actual, sample_field.name)
        expected_value = getattr(expected, sample_field.name)
        if isinstance(expected_value, np.ndarray):
            assert actual_value.dtype == expected_value.dtype
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize("num_mtp_layers", [0, 3])
@pytest.mark.parametrize("per_sample_pad_multiple", [1, 8])
def test_spec_token_counts_equal_the_packed_counts(
    num_mtp_layers: int, per_sample_pad_multiple: int
) -> None:
    group = _trainable_group()
    builder = _builder()
    spec_group = _decide(builder, group, num_mtp_layers=num_mtp_layers)
    specs = spec_group.training_sample_specs
    samples = builder.materialize(training_sample_specs=specs, rollout_group=group)
    packer = MicrobatchPacker(
        num_tokens_per_microbatch_per_dp_rank=64,
        max_context_length=64,
        pad_id=0,
        temperature=1.0,
        per_sample_pad_multiple=per_sample_pad_multiple,
        num_mtp_layers=num_mtp_layers,
        build_mrope_positions=False,
        vision_processor=VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
    )

    # Rollouts 0 and 1 one sample each, rollouts 2 and 3 two each, rollout 4's sample dropped,
    # rollout 6's first and last (its middle one dropped).
    assert [spec.rollout_id.rollout_id for spec in specs] == [0, 1, 2, 2, 3, 3, 6, 6]
    assert [spec.rollout_id.turn_id for spec in specs[-2:]] == [0, 3]
    for spec, sample in zip(specs, samples, strict=True):
        assert spec.rollout_id == sample.rollout_id
        assert spec.num_tokens == len(sample.token_ids)
        assert (spec.images, spec.image_placeholders) == (
            sample.images,
            sample.image_placeholders,
        )
        assert spec.num_response_tokens == int(sample.loss_mask[1:].sum())
        microbatch = packer.pack([sample])
        assert list(spec.loss_token_counts) == microbatch.loss_token_counts.tolist()
        assert list(spec.routing_token_counts) == (
            microbatch.routing_token_counts.tolist()
        )
    # The samples are the planned ones among every sample of each full rollout (rollout 5 is
    # not buildable), in spec order.
    planned = {spec.rollout_id for spec in specs}
    expected = [
        sample
        for rollout in group.rollouts
        if rollout.rollout_id != 5
        for sample in rollout_to_training_samples(rollout)
        if sample.rollout_id in planned
    ]
    for sample, expected_sample in zip(samples, expected, strict=True):
        _assert_samples_equal(sample, expected_sample)


@pytest.mark.parametrize("dp_degree", [1, 2])
def test_plan_global_counts_equal_the_packed_microbatches(dp_degree: int) -> None:
    batcher = Batcher.Config(num_mtp_layers=3, per_sample_pad_multiple=8).build(
        num_tokens_per_microbatch_per_dp_rank=32,
        max_context_length=32,
        num_prompts_per_train_step=1,
        dp_degree=dp_degree,
        pad_id=0,
        temperature=1.0,
        vision_processor=VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
    )
    batch, _ = batcher.add_group(rollout_group=_trainable_group())

    assert batch is not None
    microbatches = [
        microbatch
        for group in batch.microbatch_groups
        for row in group
        for microbatch in row
    ]
    assert torch.equal(
        batch.global_loss_token_counts,
        torch.stack([microbatch.loss_token_counts for microbatch in microbatches]).sum(
            0
        ),
    )
    assert torch.equal(
        batch.global_routing_token_counts,
        torch.stack(
            [microbatch.routing_token_counts for microbatch in microbatches]
        ).sum(0),
    )
    assert batch.global_loss_token_counts.dtype == torch.int64


def test_decide_reads_lengths_and_keeps_the_filters_and_metrics() -> None:
    builder = _builder()
    group = _trainable_group()

    # Without its arrays, with the full group's lengths: the same specs and metrics.
    spec_group = builder.decide(
        rollout_group=_without_arrays(group),
        turn_lengths=tuple(map(rollout_turn_lengths, group.rollouts)),
        num_mtp_layers=0,
    )

    assert spec_group.training_sample_specs == [
        dataclasses.replace(
            spec,
            images=tuple(dataclasses.replace(image, data=b"") for image in spec.images),
        )
        for spec in _decide(builder, group).training_sample_specs
    ]

    assert m.MetricsProcessor._aggregate_metrics(spec_group.metrics) == {
        "rollout/x/mean": 1.0,
        "rollout_reward/group_zero_std_frac/mean": 0.0,
        "training_sample_builder/num_rollouts_dropped_empty_completion/sum": 1.0,
        "training_sample_builder/num_samples_dropped_no_valid_tokens/sum": 2.0,
        "training_sample_builder/num_training_samples/sum": 8.0,
        **m.MetricsProcessor._aggregate_metrics(
            [m.Metric("advantage", m.SummaryStats.from_list([-1.0, 0, 1, 2, 3, 4, 5]))]
        ),
        "rollout/branches_per_rollout/mean": 10 / 6,
        "rollout/branches_per_rollout/max": 3.0,
        "rollout/min_policy_version/min": 0.0,
        "rollout/max_policy_version/max": 4.0,
    }


def test_decide_drops_failed_untrainable_and_zero_std_groups() -> None:
    builder = _builder()

    def decide(group: RolloutGroup) -> list[str]:
        spec_group = _decide(builder, group)
        assert spec_group.training_sample_specs == []
        return sorted(
            key
            for key in m.MetricsProcessor._aggregate_metrics(spec_group.metrics)
            if key.startswith("training_sample_builder/")
        )

    assert decide(RolloutGroup(group_id=1, rollouts=[])) == []
    # A first-turn prompt too long to generate from: the whole group is untrainable.
    untrainable = RolloutGroup(
        group_id=2, rollouts=[_rollout(0, [([1, 2], [-0.1])]), _rollout(1, [([1], [])])]
    )
    assert decide(untrainable) == [
        "training_sample_builder/num_groups_dropped_untrainable/sum"
    ]
    zero_std = RolloutGroup(
        group_id=3,
        rollouts=[
            _rollout(rollout_id, [([1], [-0.1])], reward=1.0) for rollout_id in (0, 1)
        ],
    )
    assert decide(zero_std) == [
        "training_sample_builder/num_groups_dropped_zero_std/sum"
    ]


class _MaskedSampleBuilder(TrainingSampleBuilder):
    """Builds samples that mask their last completion token, which the specs count."""

    def _build_training_sample(self, spec, rollout, **kwargs) -> TrainingSample:
        sample = super()._build_training_sample(spec, rollout, **kwargs)
        loss_mask = sample.loss_mask.copy()
        loss_mask[-1] = False
        return dataclasses.replace(sample, loss_mask=loss_mask)


class _LongerSampleBuilder(TrainingSampleBuilder):
    """Builds samples one masked token longer than their specs."""

    def _build_training_sample(self, spec, rollout, **kwargs) -> TrainingSample:
        sample = super()._build_training_sample(spec, rollout, **kwargs)
        return dataclasses.replace(
            sample,
            token_ids=np.append(sample.token_ids, 0),
            loss_mask=np.append(sample.loss_mask, False),
            logprobs=np.append(sample.logprobs, np.float32(0)),
            advantage=np.append(sample.advantage, np.float32(0)),
        )


@pytest.mark.parametrize(
    "builder_class",
    [_MaskedSampleBuilder, _LongerSampleBuilder],
    ids=["mask", "length"],
)
def test_packing_raises_when_a_sample_disagrees_with_its_spec(builder_class) -> None:
    """A sample whose loss mask drifts from its spec would silently rescale the loss, whose
    divisor is the specs' counts summed, and a longer one overfills the microbatch the specs
    planned (here, exactly full); packing raises a `ValueError` naming the two methods."""
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=5,
        max_context_length=5,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    group = RolloutGroup(
        group_id=0,
        rollouts=[
            _rollout(0, [([1, 2], [-0.1, -0.2])], reward=0.0),
            _rollout(1, [([1, 2], [-0.3])], reward=1.0),
        ],
    )
    batcher._training_sample_builder = builder_class(TrainingSampleBuilder.Config())

    with pytest.raises(ValueError, match="rollout_to_training_sample_specs"):
        batcher.add_group(rollout_group=group)


def test_the_step_trigger_frees_each_groups_arrays_before_the_next_materializes() -> None:
    """The step's arrays and their samples (copies) are never all alive at once: each group's
    arrays go as soon as its samples exist."""
    num_groups = 3
    array_refs: list[weakref.ref] = []
    num_alive_at_materialize = []

    class _CountingBuilder(TrainingSampleBuilder):
        def materialize(self, *, training_sample_specs, rollout_group):
            num_alive_at_materialize.append(
                sum(array_ref() is not None for array_ref in array_refs)
            )
            return super().materialize(
                training_sample_specs=training_sample_specs,
                rollout_group=rollout_group,
            )

    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=64,
        max_context_length=64,
        num_prompts_per_train_step=num_groups,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batcher._training_sample_builder = _CountingBuilder(TrainingSampleBuilder.Config())
    for group_id in range(num_groups):
        group = RolloutGroup(
            group_id=group_id,
            rollouts=[
                _rollout(0, [([1, 2], [-0.1, -0.2])], group_id=group_id, reward=0.0),
                _rollout(1, [([1, 2], [-0.3])], group_id=group_id, reward=1.0),
            ],
        )
        array_refs.append(weakref.ref(group.rollouts[0].turns[0].completion_logprobs))
        batch, _ = batcher.add_group(rollout_group=group)
        del group

    assert batch is not None
    # Groups materialize in id order; each one's arrays are gone before the next starts.
    assert num_alive_at_materialize == [3, 2, 1]


@pytest.mark.parametrize(
    "prompt_prefix_len, prompt_delta_token_ids, match",
    [
        # The full prompt as the delta: `split_prompt` stores (3, [3]).
        (0, [1, 2, 200, 3], "could reuse more"),
        # Past the previous turn's end: its prompt + completion has 3 tokens.
        (9, [3], "previous turn has only 3"),
    ],
)
def test_add_group_checks_the_prompt_prefixes_of_rollouts_it_drops(
    prompt_prefix_len: int, prompt_delta_token_ids: list[int], match: str
) -> None:
    """The prompt prefix checks run on every rollout of a group before any filter: here on a
    zero-std group, which trains nothing, whose rollout reuses fewer tokens than it could or
    more than there are."""
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=64,
        max_context_length=64,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    malformed = _rollout(0, [([1, 2], [-0.1]), ([1, 2, 200, 3], [-0.2])], reward=1.0)
    malformed.turns[1].prompt_prefix_len = prompt_prefix_len
    malformed.turns[1].prompt_delta_token_ids = np.array(
        prompt_delta_token_ids, dtype=np.int32
    )
    group = RolloutGroup(
        group_id=0, rollouts=[malformed, _rollout(1, [([1, 2], [-0.1])], reward=1.0)]
    )

    with pytest.raises(ValueError, match=match):
        batcher.add_group(rollout_group=group)
