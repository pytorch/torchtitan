# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flux-specific Grain processing and dataset recipes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast, Protocol, TypeAlias

import grain.python as grain
import numpy as np
import PIL.Image
import torch
from torch.utils.data import default_collate

from torchtitan.components.data.collators import Collator, TrainerBatch
from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    DatasetConfig,
    SampleProcessor,
    SingleDatasetConfig,
)
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.sources import HuggingFaceStreamingSource


FluxTrainingRow: TypeAlias = tuple[dict[str, Any], torch.Tensor]

_VALIDATION_TIMESTEPS = tuple((index + 0.5) / 8 for index in range(8))


class _FluxTokenizer(Protocol):
    def encode(self, text: str) -> Mapping[str, torch.Tensor]:
        ...


def _process_flux_image(
    image: PIL.Image.Image,
    *,
    output_size: int,
    rng: np.random.Generator,
) -> torch.Tensor | None:
    """Resize and randomly crop one image using the Grain-provided RNG."""
    width, height = image.size
    if width < output_size or height < output_size:
        return None

    if width >= height:
        resized_width = math.ceil(output_size / height * width)
        image = image.resize((resized_width, output_size))
        left = int(rng.integers(0, resized_width - output_size + 1))
        image = image.crop((left, 0, left + output_size, output_size))
    else:
        resized_height = math.ceil(output_size / width * height)
        image = image.resize((output_size, resized_height))
        top = int(rng.integers(0, resized_height - output_size + 1))
        image = image.crop((0, top, output_size, top + output_size))

    if image.mode != "RGB":
        image = image.convert("RGB")
    image_CHW = np.array(image, copy=True).transpose((2, 0, 1))
    return (
        torch.from_numpy(image_CHW)
        .to(dtype=torch.float32)
        .div_(255.0)
        .mul_(2.0)
        .sub_(1.0)
    )


def _validate_flux_tokens(tokens: Mapping[str, torch.Tensor]) -> None:
    missing = {"t5", "clip"} - tokens.keys()
    if missing:
        raise ValueError(f"Flux tokenizer output is missing keys: {sorted(missing)}")
    for name in ("t5", "clip"):
        if not isinstance(tokens[name], torch.Tensor):
            raise ValueError(
                f"Flux tokenizer output {name!r} must be a tensor, "
                f"got {type(tokens[name]).__name__}"
            )


class FluxSampleProcessor(SampleProcessor):
    """Crop, tokenize, and apply classifier-free prompt dropout to one row."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        image_field: str = "jpg"
        caption_field: str = "txt"
        image_size: int = 256
        prompt_dropout_prob: float = 0.0

        def __post_init__(self) -> None:
            if self.image_size <= 0:
                raise ValueError(f"image_size must be positive, got {self.image_size}")
            if not 0.0 <= self.prompt_dropout_prob <= 1.0:
                raise ValueError(
                    "prompt_dropout_prob must be in [0, 1], "
                    f"got {self.prompt_dropout_prob}"
                )

    def __init__(
        self,
        config: Config,
        *,
        runtime: DataRuntime,
    ) -> None:
        self._image_field = config.image_field
        self._caption_field = config.caption_field
        self._image_size = config.image_size
        self._prompt_dropout_prob = config.prompt_dropout_prob
        self._tokenizer = cast(_FluxTokenizer, runtime.tokenizer)
        empty_tokens = self._tokenizer.encode("")
        _validate_flux_tokens(empty_tokens)
        self._empty_tokens = empty_tokens

    def __call__(
        self,
        sample: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> FluxTrainingRow | None:
        image = sample[self._image_field]
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(
                f"Flux image must be PIL.Image.Image, got {type(image).__name__}"
            )

        prompt = sample[self._caption_field]
        if isinstance(prompt, list):
            if not prompt:
                raise ValueError("Flux caption list must not be empty")
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise ValueError(
                f"Flux caption must be a string, got {type(prompt).__name__}"
            )

        image_tensor = _process_flux_image(
            image,
            output_size=self._image_size,
            rng=rng,
        )
        if image_tensor is None:
            return None

        tokens = self._tokenizer.encode(prompt)
        _validate_flux_tokens(tokens)
        t5_tokens = tokens["t5"]
        clip_tokens = tokens["clip"]
        if rng.random() < self._prompt_dropout_prob:
            t5_tokens = self._empty_tokens["t5"]
        if rng.random() < self._prompt_dropout_prob:
            clip_tokens = self._empty_tokens["clip"]

        return (
            {
                "t5": t5_tokens,
                "clip": clip_tokens,
                "prompt": prompt,
            },
            image_tensor,
        )


class FluxCollator(Collator):
    """Collates Flux rows into the trainer's ``(inputs, labels)`` contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        del config, runtime

    def __call__(self, rows: Sequence[FluxTrainingRow]) -> TrainerBatch:
        return default_collate(list(rows))


def _add_validation_timestep(
    index: int,
    sample: FluxTrainingRow,
) -> FluxTrainingRow:
    inputs, labels = sample
    inputs = dict(inputs)
    inputs["timestep"] = _VALIDATION_TIMESTEPS[index % len(_VALIDATION_TIMESTEPS)]
    return inputs, labels


@dataclass(frozen=True, kw_only=True, slots=True)
class FluxValidationTimestepConfig:
    """Adds checkpointable round-robin validation timesteps to Flux rows."""

    dataset: DatasetConfig
    generate_timesteps: bool = True

    def __post_init__(self) -> None:
        if (
            self.generate_timesteps
            and isinstance(self.dataset, SingleDatasetConfig)
            and isinstance(self.dataset.process, FluxSampleProcessor.Config)
            and self.dataset.process.prompt_dropout_prob != 0.0
        ):
            raise ValueError(
                "prompt_dropout_prob must be 0.0 when generate_timesteps=True"
            )

    def build(
        self,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.MapDataset | grain.IterDataset:
        dataset = self.dataset.build(runtime=runtime, options=options)
        if not self.generate_timesteps:
            return dataset
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=runtime.read_options)
        return dataset.map_with_index(_add_validation_timestep)


def _cc12m_source() -> HuggingFaceStreamingSource.Config:
    return HuggingFaceStreamingSource.Config(
        path="pixparse/cc12m-wds",
        load_dataset_kwargs={"split": "train"},
    )


def _cc12m_test_source() -> HuggingFaceStreamingSource.Config:
    return HuggingFaceStreamingSource.Config(
        path="tests/assets/cc12m_test",
        load_dataset_kwargs={
            "split": "train",
            "data_files": {"train": "*.tar"},
        },
    )


def _coco_validation_source() -> HuggingFaceStreamingSource.Config:
    return HuggingFaceStreamingSource.Config(
        path="howard-hou/COCO-Text",
        load_dataset_kwargs={"split": "validation"},
    )


_FLUX_SOURCES = {
    "cc12m-wds": _cc12m_source,
    "cc12m-test": _cc12m_test_source,
    "coco-validation": _coco_validation_source,
}


def flux_dataset_config(
    dataset_name: str,
    *,
    image_size: int = 256,
    prompt_dropout_prob: float = 0.0,
) -> SingleDatasetConfig:
    """Create a Flux source recipe with its source-specific field names."""
    try:
        source_factory = _FLUX_SOURCES[dataset_name.lower()]
    except KeyError as error:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {sorted(_FLUX_SOURCES)}"
        ) from error

    if dataset_name.lower() == "coco-validation":
        image_field, caption_field = "image", "caption"
    else:
        image_field, caption_field = "jpg", "txt"

    return SingleDatasetConfig(
        source=source_factory(),
        process=FluxSampleProcessor.Config(
            image_field=image_field,
            caption_field=caption_field,
            image_size=image_size,
            prompt_dropout_prob=prompt_dropout_prob,
        ),
        filters=(_is_valid_flux_row,),
    )


def flux_validation_dataset_config(
    dataset_name: str = "coco-validation",
    *,
    image_size: int = 256,
    generate_timesteps: bool = True,
) -> FluxValidationTimestepConfig:
    return FluxValidationTimestepConfig(
        dataset=flux_dataset_config(
            dataset_name,
            image_size=image_size,
            prompt_dropout_prob=0.0,
        ),
        generate_timesteps=generate_timesteps,
    )


def _is_valid_flux_row(sample: FluxTrainingRow | None) -> bool:
    return sample is not None


def flux_image_size(dataset: DatasetConfig) -> int:
    """Return the image size bound by a Flux dataset processor."""
    if isinstance(dataset, FluxValidationTimestepConfig):
        dataset = dataset.dataset
    if not isinstance(dataset, SingleDatasetConfig):
        raise ValueError("Flux dataloader dataset must be a SingleDatasetConfig")
    process = dataset.process
    if not isinstance(process, FluxSampleProcessor.Config):
        raise ValueError("Flux dataloader dataset must use FluxSampleProcessor.Config")
    return process.image_size


def flux_validation_loader_config(
    *,
    dataset_name: str = "coco-validation",
    image_size: int = 256,
) -> GrainDataLoader.Config:
    """Build the generic Grain loader recipe used by Flux validation."""
    return GrainDataLoader.Config(
        dataset=flux_validation_dataset_config(
            dataset_name,
            image_size=image_size,
            generate_timesteps=True,
        ),
        collator=FluxCollator.Config(),
    )


def with_flux_validation_timesteps(
    config: GrainDataLoader.Config,
    *,
    generate_timesteps: bool,
) -> GrainDataLoader.Config:
    """Return a loader recipe with the requested validation timestep behavior."""
    dataset = config.dataset
    if isinstance(dataset, FluxValidationTimestepConfig):
        dataset = dataset.dataset
    return replace(
        config,
        dataset=FluxValidationTimestepConfig(
            dataset=dataset,
            generate_timesteps=generate_timesteps,
        ),
    )


__all__ = [
    "FluxCollator",
    "FluxSampleProcessor",
    "FluxTrainingRow",
    "FluxValidationTimestepConfig",
    "_VALIDATION_TIMESTEPS",
    "flux_dataset_config",
    "flux_image_size",
    "flux_validation_dataset_config",
    "flux_validation_loader_config",
    "with_flux_validation_timesteps",
]
