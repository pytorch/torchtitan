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
from PIL import Image

from torch.distributed.checkpoint.stateful import Stateful

from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader

from torchtitan.config_manager import JobConfig
from torchtitan.experiments.flux.model.modules.hf_embedder import HFEmbedder
from torchtitan.tools.logging import logger
from torchvision import transforms


def _process_cc12m_image(
    img: Image.Image,
    output_size: int = 256,
) -> Image.Image:
    """Process CC12M image to the desired size."""

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
    return resized_img


def _flux_data_processor(
    sample: dict[str, Any],
    t5_encoder: HFEmbedder,
    clip_encoder: HFEmbedder,
    output_size: int = 256,
) -> dict[str, Any]:
    """
    Preprocess CC12M dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_encoder: T5 encoder
        clip_encoder: CLIP encoder
        output_size: The output image size

    """

    img = _process_cc12m_image(sample["jpg"], output_size=output_size)
    img = transforms.ToTensor()(img)
    t5_tokens = t5_encoder(sample["txt"])
    clip_tokens = clip_encoder(sample["txt"])

    return {
        "image": img,
        "clip_encodings": clip_tokens.squeeze(0),
        "t5_encodings": t5_tokens.squeeze(0),
    }


@dataclass
class TextToImageDatasetConfig:
    path: str
    loader: Callable
    data_processor: Callable


DATASETS = {
    "cc12m": TextToImageDatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        data_processor=_flux_data_processor,
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


class FluxDataset(IterableDataset, Stateful):
    """Dataset for FLUX text-to-image model.

    Args:
    dataset_name (str): Name of the dataset.
    dataset_path (str): Path to the dataset.
    model_transform (Transform): Callable that applies model-specific preprocessing to the sample.
    dp_rank (int): Data parallel rank.
    dp_world_size (int): Data parallel world size.
    infinite (bool): Whether to loop over the dataset infinitely.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        t5_encoder: HFEmbedder,
        clip_encoder: HFEmbedder,
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

        # TODO(jianiw): put the two encoders and data in the same device
        self._t5_encoder = t5_encoder
        self._clip_encoder = clip_encoder
        self._data_processor = data_processor

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

                # Use the dataset-specific preprocessor
                sample_dict = self._data_processor(
                    sample, self._t5_encoder, self._clip_encoder, output_size=256
                )

                self._all_samples.extend(sample_dict)

                yield sample_dict

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_samples = state_dict["all_samples"]

    def state_dict(self):
        return {
            "all_samples": self._all_samples,
            "sample_idx": self._sample_idx,
        }


def _load_t5(t5_encoder_name, device: str = "cpu", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder(
        t5_encoder_name,
        max_length=max_length,
        torch_dtype=torch.bfloat16,
    ).to(device)


def _load_clip(clip_encoder_name, device: str = "cpu") -> HFEmbedder:
    # The max length is set to be 77
    return HFEmbedder(clip_encoder_name, max_length=77, torch_dtype=torch.bfloat16).to(
        device
    )


def build_flux_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size

    t5_encoder_name = job_config.encoder.t5_encoder
    clip_encoder_name = job_config.encoder.clip_encoder
    encoder_device = job_config.encoder.encoder_device  # "cuda" or "cpu"
    max_encoding_len = job_config.encoder.max_encoding_len

    t5_encoder = _load_t5(
        t5_encoder_name, device=encoder_device, max_length=max_encoding_len
    )
    clip_encoder = _load_clip(clip_encoder_name, device=encoder_device)

    ds = FluxDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        t5_encoder=t5_encoder,
        clip_encoder=clip_encoder,
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
