# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import grain.python as grain
import numpy as np
import PIL.Image
import torch

from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    SingleDatasetConfig,
)
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.models.flux.flux_datasets import (
    FluxCollator,
    FluxSampleProcessor,
    FluxValidationTimestepConfig,
)


class FakeTokenizer:
    def encode(self, text: str) -> dict[str, torch.Tensor]:
        value = len(text)
        return {
            "t5": torch.full((5,), value, dtype=torch.long),
            "clip": torch.full((3,), value + 100, dtype=torch.long),
        }


@dataclass(frozen=True)
class RowsSourceConfig:
    rows: tuple[dict, ...]

    def build(self, **_):
        return self.rows


def _image(width: int, height: int, offset: int) -> PIL.Image.Image:
    y, x = np.mgrid[:height, :width]
    pixels = np.stack(
        (
            (x * 17 + offset) % 256,
            (y * 29 + offset * 3) % 256,
            ((x + y) * 11 + offset * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return PIL.Image.fromarray(pixels, mode="RGB")


def _rows(count: int, *, image_size: int = 8) -> tuple[dict, ...]:
    return tuple(
        {
            "image": _image(
                image_size + 11 if index % 2 == 0 else image_size,
                image_size if index % 2 == 0 else image_size + 9,
                index,
            ),
            "prompt": f"sample-{index}",
        }
        for index in range(count)
    )


def _runtime(batch_size: int = 2) -> DataRuntime:
    return DataRuntime(
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=batch_size,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )


def _options(*, seed: int = 17, repeat: bool = True) -> BuildOptions:
    return BuildOptions(
        seed=seed,
        shuffle=False,
        repeat=repeat,
        dp_rank=0,
        dp_world_size=1,
    )


def _dataset(
    rows: tuple[dict, ...],
    *,
    image_size: int = 8,
    dropout: float = 0.0,
) -> SingleDatasetConfig:
    return SingleDatasetConfig(
        source=RowsSourceConfig(rows),
        process=FluxSampleProcessor.Config(
            image_field="image",
            caption_field="prompt",
            image_size=image_size,
            prompt_dropout_prob=dropout,
        ),
        filters=(lambda row: row is not None,),
    )


def _loader(
    dataset,
    *,
    batch_size: int = 2,
    seed: int = 17,
    repeat: bool = True,
):
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=FluxCollator.Config(),
        seed=seed,
        shuffle=False,
        repeat=repeat,
        batch_prefetch_buffer_size=1,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=batch_size,
    )


def test_flux_processor_uses_seeded_crop_and_prompt_dropout():
    dataset = _dataset(_rows(12), dropout=0.5)
    first = list(
        dataset.build(
            runtime=_runtime(),
            options=_options(seed=91, repeat=False),
        )
    )
    second = list(
        dataset.build(
            runtime=_runtime(),
            options=_options(seed=91, repeat=False),
        )
    )

    assert len(first) == len(second) == 12
    for actual, expected in zip(first, second, strict=True):
        assert actual[0]["prompt"] == expected[0]["prompt"]
        assert torch.equal(actual[0]["t5"], expected[0]["t5"])
        assert torch.equal(actual[0]["clip"], expected[0]["clip"])
        assert torch.equal(actual[1], expected[1])

    assert any(
        torch.equal(row[0]["t5"], torch.zeros_like(row[0]["t5"])) for row in first
    )
    assert any(not torch.equal(first[0][1], row[1]) for row in first[1:])


def test_flux_loader_checkpoint_restores_random_transform_without_reseeding():
    dataset = _dataset(_rows(20), dropout=0.5)
    loader = _loader(dataset, seed=91)
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = next(iterator)

    torch.rand(100)
    restored = _loader(dataset, seed=91)
    restored.load_state_dict(state)
    actual = next(iter(restored))

    assert actual[0]["prompt"] == expected[0]["prompt"]
    assert torch.equal(actual[0]["t5"], expected[0]["t5"])
    assert torch.equal(actual[0]["clip"], expected[0]["clip"])
    assert torch.equal(actual[1], expected[1])


def test_flux_validation_timesteps_follow_accepted_rows_and_restore():
    rows = list(_rows(8))
    rows[1] = {
        "image": _image(4, 12, 99),
        "prompt": "rejected",
    }
    dataset = FluxValidationTimestepConfig(
        dataset=_dataset(tuple(rows)),
        generate_timesteps=True,
    )
    loader = _loader(dataset, batch_size=2, repeat=True)
    iterator = iter(loader)

    first = next(iterator)
    assert torch.equal(
        first[0]["timestep"],
        torch.tensor([0.0625, 0.1875], dtype=torch.float64),
    )
    state = loader.state_dict()
    expected = next(iterator)

    restored = _loader(
        FluxValidationTimestepConfig(
            dataset=_dataset(tuple(rows)),
            generate_timesteps=True,
        ),
        batch_size=2,
        repeat=True,
    )
    restored.load_state_dict(state)
    actual = next(iter(restored))
    assert torch.equal(actual[0]["timestep"], expected[0]["timestep"])
    assert torch.equal(actual[1], expected[1])


def test_flux_collator_preserves_trainer_batch_contract():
    loader = _loader(_dataset(_rows(3)), batch_size=2, repeat=False)
    inputs, labels = next(iter(loader))

    assert set(inputs) == {"t5", "clip", "prompt"}
    assert inputs["t5"].shape == (2, 5)
    assert inputs["clip"].shape == (2, 3)
    assert inputs["prompt"] == ["sample-0", "sample-1"]
    assert labels.shape == (2, 3, 8, 8)
    assert labels.dtype == torch.float32
    assert labels.min() >= -1.0
    assert labels.max() <= 1.0
