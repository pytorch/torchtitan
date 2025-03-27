# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from einops import rearrange, repeat

from PIL.Image import Transform
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from torchvision import transforms


def _process_cc12m_image(
    sample: dict[str, Any], output_size: int = 256
) -> tuple[str, torch.Tensor]:
    """
    Preprocess CC12M dataset sample image.
    Preprocess Steps:
        1. Ignore all the images smaller than output_size * output_size.
        2. Resize and crop the image to output_size * output_size.

    Args:
        sample: A sample from dataset.
        output_size: The output image size.

    Returns:
        A tuple of (prompt, image).

    """
    img = sample["jpg"]
    prompt = sample["txt"]

    if img.width < output_size or img.height < output_size:
        logger.info("Skip dataset sample because of image is too small.")
        return prompt, None

    width, height = img.size
    if width >= height:
        # resize height to be equal to output_size, then crop
        new_width, new_height = math.ceil(output_size / height * width), output_size
        img = img.resize((new_width, new_height))
        left = random.randint(0, new_width - output_size)
        resized_img = img.crop((left, 0, left + output_size, output_size))
    else:
        # resize width to be equal to output_size, the crop
        new_width, new_height = (
            output_size,
            math.ceil(output_size / width * height),
        )
        img = img.resize((new_width, new_height))
        lower = random.randint(0, new_width - output_size)
        resized_img = img.crop((0, lower, output_size, lower + output_size))

    assert resized_img.size[0] == resized_img.size[1] == output_size
    return prompt, resized_img


@dataclass
class TextToImageDatasetConfig:
    path: str
    loader: Callable
    data_processor: Callable


DATASETS = {
    "cc12m": TextToImageDatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        data_processor=_process_cc12m_image,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: Optional[str] = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.data_processor


class TextToImageDataset(IterableDataset, Stateful):
    """Dataset for text-to-image dataset.

    Args:
    dataset_name (str): Name of the dataset.
    dataset_path (str): Path to the dataset.
    model_transform (Transform): Callable that applies model-specific preprocessing to the sample.
            See :class:`~.torchtitan.experiments.flux.FluxTransform` for an example.
    dp_rank (int): Data parallel rank.
    dp_world_size (int): Data parallel world size.
    infinite (bool): Whether to loop over the dataset infinitely.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        model_transform: Transform,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:

        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._data_processor = data_processor
        self._image_transformer = (
            transforms.ToTensor()
        )  # TODO(jianiw): merge the image transformer into the model transformer
        self._model_transformer = model_transform

        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_samples: list[dict[str, Any]] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1

                # Use the dataset-specific text processor
                sample_prompt, sample_image = self._data_processor(sample)
                if sample_prompt is None or sample_image is None:
                    continue

                self._all_target_imgs.extend(self._image_transformer(sample_image))
                self._all_prompts.extend(sample_prompt)

                yield {
                    "image": self._image_transformer(sample_image),
                    "text": sample_prompt,
                }

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_target_imgs = state_dict["img_buffer"]
        self._all_prompts = state_dict["txt_buffer"]

    def state_dict(self):
        return {
            "img_buffer": self._all_target_imgs,
            "txt_buffer": self._all_prompts,
            "sample_idx": self._sample_idx,
        }


def text_to_image_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size

    ds = TextToImageDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
