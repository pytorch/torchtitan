# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import grain.python as grain
import numpy as np
import PIL.Image
import torch
import tyro
from datasets import load_dataset
from torch.utils.data import default_collate

from torchtitan.components.data.collators import Collator, TrainerBatch
from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    DatasetConfig as GrainDatasetConfig,
    SampleProcessor,
    SingleDatasetConfig,
)
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.models.flux.tokenizer import FluxTokenizerContainer
from torchtitan.tools.logging import logger


def _process_cc12m_image(
    img: PIL.Image.Image,
    output_size: int = 256,
    *,
    rng: np.random.Generator,
) -> torch.Tensor | None:
    """Process CC12M image to the desired size."""

    width, height = img.size
    # Skip low resolution images
    if width < output_size or height < output_size:
        return None

    if width >= height:
        # resize height to be equal to output_size, then crop
        new_width, new_height = math.ceil(output_size / height * width), output_size
        img = img.resize((new_width, new_height))
        left = int(rng.integers(0, new_width - output_size + 1))
        resized_img = img.crop((left, 0, left + output_size, output_size))
    else:
        # resize width to be equal to output_size, the crop
        new_width, new_height = (
            output_size,
            math.ceil(output_size / width * height),
        )
        img = img.resize((new_width, new_height))
        lower = int(rng.integers(0, new_height - output_size + 1))
        resized_img = img.crop((0, lower, output_size, lower + output_size))

    assert resized_img.size[0] == resized_img.size[1] == output_size

    # Convert grayscale images, and RGBA, CMYK images
    if resized_img.mode != "RGB":
        resized_img = resized_img.convert("RGB")

    # Normalize the image to [-1, 1]
    np_img = np.array(resized_img).transpose((2, 0, 1))
    tensor_img = torch.tensor(np_img).float() / 255.0 * 2.0 - 1.0

    # NOTE: The following commented code is an alternative way
    # img_transform = transforms.Compose(
    #     [
    #         transforms.Resize(max(output_size, output_size)),
    #         transforms.CenterCrop((output_size, output_size)),
    #         transforms.ToTensor(),
    #     ]
    # )
    # tensor_img = img_transform(img)

    return tensor_img


def _cc12m_wds_data_processor(
    sample: dict[str, Any],
    tokenizer: FluxTokenizerContainer,
    output_size: int = 256,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Preprocess CC12M dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        tokenizer: FluxTokenizerContainer that encodes text with both T5 and CLIP
        output_size: The output image size

    """
    img = _process_cc12m_image(sample["jpg"], output_size=output_size, rng=rng)
    tokens = tokenizer.encode(sample["txt"])

    return {
        "image": img,
        **tokens,
        "prompt": sample["txt"],
    }


def _coco_data_processor(
    sample: dict[str, Any],
    tokenizer: FluxTokenizerContainer,
    output_size: int = 256,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Preprocess COCO dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        tokenizer: FluxTokenizerContainer that encodes text with both T5 and CLIP
        output_size: The output image size

    """
    img = _process_cc12m_image(sample["image"], output_size=output_size, rng=rng)
    prompt = sample["caption"]
    if isinstance(prompt, list):
        prompt = prompt[0]
    tokens = tokenizer.encode(prompt)

    return {
        "image": img,
        **tokens,
        "prompt": prompt,
    }


DATASETS = {
    "cc12m-wds": DatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_cc12m_wds_data_processor,
    ),
    "cc12m-test": DatasetConfig(
        path="tests/assets/cc12m_test",
        loader=lambda path: load_dataset(
            path, split="train", data_files={"train": "*.tar"}, streaming=True
        ),
        sample_processor=_cc12m_wds_data_processor,
    ),
    "coco-validation": DatasetConfig(
        path="howard-hou/COCO-Text",
        loader=lambda path: load_dataset(path, split="validation", streaming=True),
        sample_processor=_coco_data_processor,
    ),
}


class FluxSampleProcessor(SampleProcessor):
    """Applies an existing Flux processor and classifier-free prompt dropout."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        data_processor: Annotated[Callable, tyro.conf.Suppress]
        prompt_dropout_prob: float = 0.0
        img_size: int = 256

        def __post_init__(self) -> None:
            if not 0.0 <= self.prompt_dropout_prob <= 1.0:
                raise ValueError(
                    "prompt_dropout_prob must be in [0, 1], "
                    f"got {self.prompt_dropout_prob}"
                )

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        if not isinstance(runtime.tokenizer, FluxTokenizerContainer):
            raise ValueError(
                "Flux dataloader requires a FluxTokenizerContainer as tokenizer. "
                "Set tokenizer=FluxTokenizerContainer.Config(...) in your trainer config."
            )
        self._tokenizer = runtime.tokenizer
        empty_tokens = self._tokenizer.encode("")
        self._t5_empty_token = empty_tokens["t5"]
        self._clip_empty_token = empty_tokens["clip"]
        self._data_processor = config.data_processor
        self.prompt_dropout_prob = config.prompt_dropout_prob
        self.img_size = config.img_size

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> tuple[dict[str, Any], torch.Tensor] | None:
        # Use the dataset-specific preprocessor
        sample_dict = self._data_processor(
            sample,
            self._tokenizer,
            output_size=self.img_size,
            rng=rng,
        )

        # skip low quality image or image with color channel = 1
        if sample_dict["image"] is None:
            sample_id = sample.get("__key__", "unknown")
            logger.warning(
                f"Low quality image {sample_id} is skipped in Flux Dataloader."
            )
            return None

        # Classifier-free guidance: Replace some of the strings with empty strings.
        dropout_prob = self.prompt_dropout_prob
        if dropout_prob > 0.0:
            if rng.random() < dropout_prob:
                sample_dict["t5"] = self._t5_empty_token
            if rng.random() < dropout_prob:
                sample_dict["clip"] = self._clip_empty_token

        labels = sample_dict.pop("image")
        return sample_dict, labels


class FluxCollator(Collator):
    """Uses PyTorch's default collation for Flux rows."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        del config, runtime

    def __call__(
        self,
        rows: Sequence[tuple[dict[str, Any], torch.Tensor]],
    ) -> TrainerBatch:
        return default_collate(list(rows))


_VALIDATION_TIMESTEPS = tuple((index + 0.5) / 8 for index in range(8))


def _add_validation_timestep(
    index: int,
    sample: tuple[dict[str, Any], torch.Tensor],
) -> tuple[dict[str, Any], torch.Tensor]:
    sample_dict, labels = sample
    sample_dict = dict(sample_dict)
    sample_dict["timestep"] = _VALIDATION_TIMESTEPS[index % len(_VALIDATION_TIMESTEPS)]
    return sample_dict, labels


@dataclass(frozen=True, kw_only=True, slots=True)
class FluxValidationDatasetConfig:
    """Adds checkpointable round-robin validation timesteps to Flux rows."""

    dataset: GrainDatasetConfig
    generate_timesteps: bool = True

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


def flux_image_size(dataset: GrainDatasetConfig) -> int:
    """Return the image size configured by a Flux dataset processor."""
    if isinstance(dataset, FluxValidationDatasetConfig):
        dataset = dataset.dataset
    if not isinstance(dataset, SingleDatasetConfig):
        raise ValueError("Flux dataloader dataset must be a SingleDatasetConfig")
    process = dataset.process
    if not isinstance(process, FluxSampleProcessor.Config):
        raise ValueError("Flux dataloader dataset must use FluxSampleProcessor.Config")
    return process.img_size
