# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for batcher packing (fixed-bin LPT, FFD fallback) and image microbatches."""

import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from renderers import PlaceholderRange
from torchtitan.components.data.multimodal.mrope import get_mrope_positions
from torchtitan.components.data.multimodal.vision_processor import (
    PixelBudgetResize,
    VisionProcessor,
)
from torchtitan.components.renderer import hash_image_file
from torchtitan.models.qwen3_5 import QWEN3_5_VISION_PROCESSOR_KWARGS
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID

from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.training_sample_builder import TrainingSample


def _make_samples(lengths: list[int]) -> list[TrainingSample]:
    return [
        TrainingSample(
            min_policy_version=0,
            max_policy_version=0,
            rollout_id=RolloutTurnID(group_id=0, rollout_id=index, turn_id=0),
            token_ids=np.arange(length + 1),
            loss_mask=np.array([False] + [True] * length),
            logprobs=np.zeros(length + 1, dtype=np.float32),
            advantage=np.zeros(length + 1, dtype=np.float32),
        )
        for index, length in enumerate(lengths)
    ]


def test_padding_workload_matches_existing_full_length_segments() -> None:
    batcher = Batcher.Config(max_num_documents=20).build(
        num_tokens_per_microbatch_per_dp_rank=60,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    long_samples = _make_samples([8] * 6)
    short_samples = _make_samples([1] * 20)

    assert batcher._attention_workload(long_samples) == 6 * 8**2 + 10**2 + 2**2
    assert batcher._attention_workload(short_samples) == 20 + 4 * 10**2
    assert batcher._attention_workload([]) == 6 * 10**2

    for samples in (long_samples, short_samples, []):
        microbatch = batcher._pack_training_samples(samples)
        num_padding_segments = int(
            ((microbatch.positions == 0) & microbatch.padding_mask).sum().item()
        )
        assert num_padding_segments <= 6


def test_lpt_uses_padding_workload_for_full_length_documents() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=48,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([8] * 3 + [2] * 16)

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        batcher._attention_workload(rank_samples)
        for row in assignments
        for rank_samples in row
    ] == [288, 264]
    assert sorted(
        sum(map(batcher.num_tokens_to_pack, rank_samples))
        for row in assignments
        for rank_samples in row
    ) == [26, 30]


def test_lpt_does_not_leave_a_full_length_document_in_an_expensive_bin() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=20,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([10, 5, 5, 5, 5])

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        batcher._attention_workload(rank_samples)
        for row in assignments
        for rank_samples in row
    ] == [150, 100]


def test_lpt_rebalances_ffd_bins_with_document_limit() -> None:
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([4, 4, 3, 3, 3, 3, 2, 2])

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ] == [[26, 26], [22, 22]]
    assert all(
        len(rank_samples) <= 3
        and sum(map(batcher.num_tokens_to_pack, rank_samples)) <= 8
        for microbatch in assignments
        for rank_samples in microbatch
    )
    assert sorted(
        sample.rollout_id.rollout_id
        for microbatch in assignments
        for rank_samples in microbatch
        for sample in rank_samples
    ) == list(range(len(samples)))


def test_lpt_falls_back_when_ffd_fits_but_lpt_cannot() -> None:
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=6,
        max_context_length=3,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([3] * 8 + [2] * 9)
    assert batcher._pack_with_lpt(samples, target_num_bins=7) is None

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert len(assignments) == 7
    assert [
        sum(batcher.num_tokens_to_pack(sample) for sample in rank_samples)
        for (rank_samples,) in assignments
    ] == [6] * 7
    assert (
        sorted(len(rank_samples) for (rank_samples,) in assignments)
        == [2] * 4 + [3] * 3
    )


def test_three_pp_microbatches_balance_rank_costs_within_step() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([10, 9, 8, 7, 6, 5])

    assignments = batcher._assign_training_samples_to_microbatches(
        samples, num_pp_microbatches=3
    )

    assert len(assignments) == 3
    workloads = [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ]
    assert workloads == [[100, 82], [58, 68], [50, 52]]
    assert [sum(rank_workloads) for rank_workloads in zip(*workloads)] == [208, 202]


def test_packing_preserves_all_samples_and_capacity_constraints() -> None:
    rng = random.Random(42)
    for _ in range(40):
        seq_len = rng.randint(3, 12)
        num_rows = rng.randint(1, 3)
        dp_degree = rng.randint(1, 4)
        num_pp_microbatches = rng.choice([1, 3])
        max_num_documents = rng.choice([None, 1, 3, 6])
        batcher = Batcher.Config(max_num_documents=max_num_documents).build(
            num_tokens_per_microbatch_per_dp_rank=num_rows * seq_len,
            max_context_length=seq_len,
            num_prompts_per_train_step=1,
            dp_degree=dp_degree,
            pad_id=0,
            temperature=1.0,
        )
        samples = _make_samples(
            [rng.randint(1, seq_len) for _ in range(rng.randint(1, 30))]
        )

        assignments = batcher._assign_training_samples_to_microbatches(
            samples, num_pp_microbatches=num_pp_microbatches
        )

        assert len(assignments) % num_pp_microbatches == 0
        assert all(len(row) == dp_degree for row in assignments)
        for row in assignments:
            for rank_samples in row:
                assert sum(map(batcher.num_tokens_to_pack, rank_samples)) <= (
                    num_rows * seq_len
                )
                if max_num_documents is not None:
                    assert len(rank_samples) <= max_num_documents
        assert sorted(
            sample.rollout_id.rollout_id
            for row in assignments
            for rank_samples in row
            for sample in rank_samples
        ) == list(range(len(samples)))


def test_microbatch_carries_the_sampling_temperature_into_the_loss() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=0.5,
    )
    microbatch = batcher._pack_training_samples(_make_samples([3]))

    temperature = microbatch.as_input_dict()["loss_kwargs"]["temperature"]
    assert temperature.dtype == torch.float32
    assert temperature.tolist() == [0.5] * 8


def test_microbatch_moves_loss_kwargs_with_the_model_inputs() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    microbatch = batcher._pack_training_samples(_make_samples([3]))

    input_dict = microbatch.to_input_dict("meta")

    assert input_dict["input"].device.type == "meta"
    assert set(input_dict["loss_kwargs"]) == {
        "generator_logprobs",
        "temperature",
        "loss_mask",
        "advantages",
    }
    for value in input_dict["loss_kwargs"].values():
        assert value.device.type == "meta"


_VISION_PROCESSOR = dataclasses.replace(
    VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
    resize=PixelBudgetResize.Config(min_pixels=32 * 32, max_pixels=64 * 64),
)


@pytest.fixture
def image_paths(tmp_path) -> list[str]:
    rng = np.random.default_rng(0)
    paths = []
    for name, (width, height) in (("a.png", (64, 32)), ("b.png", (32, 64))):
        path = tmp_path / name
        Image.fromarray(
            rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        ).save(path)
        paths.append(str(path))
    return paths


def _prompt_image(path: str, offset: int) -> tuple[PromptImage, PlaceholderRange]:
    """The image and placeholders the image-reference renderer gives the file at ``path``."""
    processor = _VISION_PROCESSOR.build()
    data = Path(path).read_bytes()
    width, height = Image.open(path).size
    grid_thw = processor.grid_thw(height, width)
    image = PromptImage(
        hash_value=hash_image_file(data),
        source_path=path,
        grid_thw=grid_thw,
        data=data,
    )
    length = processor.num_placeholder_tokens(grid_thw)
    return image, PlaceholderRange(offset=offset, length=length)


def _token_grid(image: PromptImage) -> tuple[int, int, int]:
    """``image``'s grid after the 2x2 spatial merge, as `get_mrope_positions` takes it."""
    t, h, w = image.grid_thw
    return t, h // 2, w // 2


def _image_sample(
    index: int, images: list[tuple[PromptImage, PlaceholderRange]], length: int
) -> TrainingSample:
    [sample] = _make_samples([length])
    return dataclasses.replace(
        sample,
        rollout_id=RolloutTurnID(group_id=0, rollout_id=index, turn_id=0),
        images=tuple(image for image, _ in images),
        image_placeholders=tuple(placeholder for _, placeholder in images),
    )


def test_microbatch_encodes_each_distinct_image_once(image_paths) -> None:
    batcher = Batcher.Config(build_mrope_positions=True).build(
        num_tokens_per_microbatch_per_dp_rank=64,
        max_context_length=32,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
        vision_processor=_VISION_PROCESSOR,
    )
    image_a, placeholder_a = _prompt_image(image_paths[0], offset=1)
    image_b, placeholder_b = _prompt_image(
        image_paths[1], offset=2 + placeholder_a.length
    )
    # Two samples of one group show image a; the second also shows image b; a third is text.
    samples = [
        _image_sample(0, [(image_a, placeholder_a)], 12),
        _image_sample(1, [(image_a, placeholder_a), (image_b, placeholder_b)], 16),
        _image_sample(2, [], 5),
    ]

    microbatch = batcher._pack_training_samples(samples)
    kwargs = microbatch.model_kwargs

    # The batcher ships the image files once each; moving the inputs to the trainer's
    # device turns them into the model's pixel_values.
    processor = _VISION_PROCESSOR.build()
    pixel_values = torch.cat(
        [
            processor.patchify(processor.resize_and_normalize(image.data)[0])[0]
            for image in (image_a, image_b)
        ]
    )
    grid_thw = torch.tensor([image_a.grid_thw, image_b.grid_thw])
    assert bytes(kwargs["image_files"].numpy()) == image_a.data + image_b.data
    assert kwargs["image_file_sizes"].tolist() == [
        len(image_a.data),
        len(image_b.data),
    ]
    input_dict = microbatch.preprocess_images().as_input_dict()
    assert "image_files" not in input_dict and "image_file_sizes" not in input_dict
    assert torch.equal(input_dict["pixel_values"], pixel_values)
    assert torch.equal(kwargs["grid_thw"], grid_thw)
    rows = kwargs["image_vision_bank_indices_T"].tolist()
    a_rows = list(range(placeholder_a.length))
    b_rows = list(
        range(placeholder_a.length, placeholder_a.length + placeholder_b.length)
    )
    expected = [-1] * 64
    expected[1 : 1 + placeholder_a.length] = a_rows
    expected[13 : 13 + placeholder_a.length] = a_rows
    start_b = 12 + placeholder_b.offset
    expected[start_b : start_b + placeholder_b.length] = b_rows
    assert rows == expected
    expected_mrope = torch.cat(
        [
            get_mrope_positions(12, [(1, _token_grid(image_a))]),
            get_mrope_positions(
                16,
                [
                    (1, _token_grid(image_a)),
                    (placeholder_b.offset, _token_grid(image_b)),
                ],
            ),
            torch.arange(5).unsqueeze(1).expand(5, 3),
            (torch.arange(64 - 33) % 32).unsqueeze(1).expand(64 - 33, 3),
        ]
    )
    assert torch.equal(kwargs["mrope_positions"], expected_mrope)


def _image_batcher(
    num_tokens: int = 16, *, build_mrope_positions: bool = True
) -> Batcher:
    return Batcher.Config(build_mrope_positions=build_mrope_positions).build(
        num_tokens_per_microbatch_per_dp_rank=num_tokens,
        max_context_length=num_tokens,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
        vision_processor=_VISION_PROCESSOR,
    )


def test_text_microbatch_with_a_vision_processor_has_no_image_inputs() -> None:
    batcher = _image_batcher()

    microbatch = batcher._pack_training_samples(_make_samples([5]))

    # The model uses the 1D positions and runs its dummy vision forward, as on every
    # rank without images.
    assert microbatch.model_kwargs == {}


def test_image_microbatch_without_mrope_positions(image_paths) -> None:
    """Without build_mrope_positions (a model whose image tokens take 1D positions, e.g.
    Muse Spark), an image microbatch carries no mrope_positions: the model uses positions."""
    image, placeholder = _prompt_image(image_paths[0], offset=1)
    batcher = _image_batcher(num_tokens=64, build_mrope_positions=False)

    microbatch = batcher._pack_training_samples(
        [_image_sample(0, [(image, placeholder)], 12)]
    )

    assert "mrope_positions" not in microbatch.model_kwargs
    assert "image_vision_bank_indices_T" in microbatch.model_kwargs


def test_image_grid_must_match_the_resized_image(image_paths) -> None:
    image, placeholder = _prompt_image(image_paths[0], offset=1)
    t, h, w = image.grid_thw
    # The renderer had another vision processor: another grid (and its placeholder
    # count) for the same image.
    image = dataclasses.replace(image, grid_thw=(t, h + 2, w))
    placeholder = PlaceholderRange(offset=1, length=t * (h + 2) * w // 4)
    batcher = _image_batcher(num_tokens=64)
    microbatch = batcher._pack_training_samples(
        [_image_sample(0, [(image, placeholder)], 12)]
    )

    # The trainer resizes the file and finds the batcher's vision processor's grid.
    with pytest.raises(ValueError, match="same vision processor"):
        microbatch.preprocess_images()


def test_placeholder_count_must_match_the_grid(image_paths) -> None:
    image, placeholder = _prompt_image(image_paths[0], offset=1)
    # Embedding rows follow the placeholder count, the vision encoder's embeddings the grid.
    placeholder = PlaceholderRange(offset=1, length=placeholder.length - 1)
    batcher = _image_batcher(num_tokens=64)

    with pytest.raises(ValueError, match="placeholders"):
        batcher._pack_training_samples([_image_sample(0, [(image, placeholder)], 12)])


def test_packing_keeps_samples_of_an_image_together(image_paths) -> None:
    # Two groups of three samples, one image per group: each group fits one microbatch.
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=64,
        max_context_length=64,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
        vision_processor=_VISION_PROCESSOR,
    )
    images = [_prompt_image(path, offset=1) for path in image_paths]
    samples = []
    for group, image in enumerate(images):
        samples += [
            _image_sample(3 * group + index, [image], 18 + index * 2 - group)
            for index in range(3)
        ]

    [row] = batcher._assign_training_samples_to_microbatches(samples)

    hashes_per_microbatch = sorted(
        sorted({image.hash_value for sample in bin_ for image in sample.images})
        for bin_ in row
    )
    assert hashes_per_microbatch == sorted([[image.hash_value] for image, _ in images])
