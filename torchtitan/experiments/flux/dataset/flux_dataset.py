# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import math
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import PIL

import torch
from datasets import Dataset, load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node

from PIL import ImageFile

from torch.distributed.checkpoint.stateful import Stateful

from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.experiments.flux.dataset.tokenizer import (
    build_flux_tokenizer,
    FluxTokenizer,
)

from torchtitan.tools.logging import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _process_cc12m_image(
    img: PIL.Image.Image,
    output_size: int = 256,
    skip_low_resolution: bool = True,
) -> Optional[torch.Tensor]:
    """Process CC12M image to the desired size."""

    width, height = img.size
    # Skip low resolution images
    if skip_low_resolution and (width < output_size or height < output_size):
        return None

    if height != output_size or width != output_size:
        if width >= height:
            # resize height to be equal to output_size, then crop
            new_width, new_height = math.ceil(output_size / height * width), output_size
            img = img.resize((new_width, new_height))
            left = torch.randint(0, new_width - output_size + 1, (1,)).item()
            resized_img = img.crop((left, 0, left + output_size, output_size))
        else:
            # resize width to be equal to output_size, the crop
            new_width, new_height = (
                output_size,
                math.ceil(output_size / width * height),
            )
            img = img.resize((new_width, new_height))
            lower = torch.randint(0, new_height - output_size + 1, (1,)).item()
            resized_img = img.crop((0, lower, output_size, lower + output_size))

        assert resized_img.size[0] == resized_img.size[1] == output_size
    else:
        resized_img = img
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
    t5_tokenizer: FluxTokenizer,
    clip_tokenizer: FluxTokenizer,
    output_size: int = 256,
    include_sample_id: bool = False,
) -> dict[str, Any]:
    """
    Preprocess CC12M dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_encoder: T5 encoder
        clip_encoder: CLIP encoder
        output_size: The output image size

    """
    img_key = "png" if "png" in sample else "jpg"
    img = _process_cc12m_image(sample[img_key], output_size=output_size)
    t5_tokens = t5_tokenizer.encode(sample["txt"])
    clip_tokens = clip_tokenizer.encode(sample["txt"])

    # Include the sample ID if available
    result = {
        "image": img,
        "clip_tokens": clip_tokens,  # type: List[int]
        "t5_tokens": t5_tokens,  # type: List[int],
        "txt": sample["txt"],
    }
    if include_sample_id:
        result["id"] = sample["__key__"]

    return result


def deserialize_numpy_array(data: bytes) -> torch.Tensor:
    """Deserialize numpy array from bytes."""
    buffer = io.BytesIO(data)

    # Load uint16 view and convert back to bf16
    uint16_data = np.load(buffer)
    return torch.from_numpy(uint16_data).view(torch.bfloat16)


def _cc12m_data_processor_from_encodings(
    sample: dict[str, Any],
    t5_tokenizer: FluxTokenizer,  # Required for API compatibility
    clip_tokenizer: FluxTokenizer,  # Required for API compatibility
    output_size: int = 256,  # Required for API compatibility
    include_sample_id: bool = False,  # Used to determine whether to include sample ID
) -> dict[str, Any]:
    """
    Process data from preprocessed encodings.

    Args:
        sample: A sample from dataset containing encodings
        t5_tokenizer: T5 tokenizer (kept for API compatibility)
        clip_tokenizer: CLIP tokenizer (kept for API compatibility)
        output_size: Output image size (kept for API compatibility)
        include_sample_id: Whether to include sample ID in the result
        job_config: Job configuration
        autoencoder: Optional autoencoder model. If provided, will be used to reconstruct images
                    from mean and logvar. If not provided, no reconstruction will be performed.

    Returns:
        Processed sample with tensors and optionally reconstructed image
    """
    # important to copy the sample as we are modifying it in place
    sample = sample.copy()

    sample["t5_encodings"] = deserialize_numpy_array(sample["t5_encodings"])
    sample["clip_encodings"] = deserialize_numpy_array(sample["clip_encodings"])
    sample["mean"] = deserialize_numpy_array(sample["mean"])
    sample["logvar"] = deserialize_numpy_array(sample["logvar"])
    sample_id = sample.pop("__key__")
    if include_sample_id:
        sample["id"] = sample_id

    return sample


@dataclass
class TextToImageDatasetConfig:
    path: str
    loader: Callable
    data_processor: Callable


DATASETS = {
    "cc12m_wds": TextToImageDatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        data_processor=_cc12m_wds_data_processor,
    ),
    "cc12m_disk": TextToImageDatasetConfig(
        path="/dataset/cc12m_disk",
        loader=lambda path: load_from_disk(path),
        data_processor=partial(_cc12m_wds_data_processor, include_sample_id=True),
    ),
    "cc12m_preprocessed": TextToImageDatasetConfig(
        path="/dataset/cc12m_preprocessed",
        loader=lambda path: load_from_disk(path),
        data_processor=_cc12m_data_processor_from_encodings,
    ),
    "cc12m_wds_30k": TextToImageDatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True).take(
            30_000
        ),
        data_processor=_cc12m_wds_data_processor,
    ),
    "cc12m_test": TextToImageDatasetConfig(
        path="torchtitan/experiments/flux/tests/assets/cc12m_test",
        loader=lambda path: load_dataset(
            path, split="train", data_files={"train": "*.tar"}, streaming=True
        ),
        data_processor=_cc12m_wds_data_processor,
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
        t5_tokenizer: Tokenizer,
        clip_tokenizer: Tokenizer,
        job_config: Optional[JobConfig] = None,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        include_sample_id: bool = False,
        subset_size: int | None = None,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)
        if subset_size is not None:
            ds = ds.take(subset_size)

        self.dataset_name = dataset_name

        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        if t5_tokenizer and clip_tokenizer:
            self._t5_tokenizer = t5_tokenizer
            self._t5_empty_token = t5_tokenizer.encode("")
            self._clip_tokenizer = clip_tokenizer
            self._clip_empty_token = clip_tokenizer.encode("")
        else:
            self._t5_tokenizer = None
            self._t5_empty_token = None
            self._clip_tokenizer = None
            self._clip_empty_token = None

        self._data_processor = data_processor
        self.job_config = job_config

        self.infinite = infinite
        self.include_sample_id = include_sample_id

        # Variables for checkpointing
        self._sample_idx = 0
        self._epoch = 0
        self._restored_checkpoint = False

    def reset(self):
        self._sample_idx = 0

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data)

    def __iter__(self):
        # Initialize the dataset iterator
        iterator = self._get_data_iter()

        # Skip samples if we're resuming from a checkpoint
        if self._restored_checkpoint:
            logger.info(f"Restoring dataset state: skipping {self._sample_idx} samples")
            for _ in range(self._sample_idx):
                next(iterator)
            self._restored_checkpoint = False

        while True:
            try:
                sample = next(iterator)
            except (UnicodeDecodeError, SyntaxError, OSError, ZeroDivisionError) as e:
                # Handle other exception, eg, dataset corruption
                logger.warning(
                    f"Dataset {self.dataset_name} has error while loading batch data. \
                    Error {type(e).__name__}: {e}. The error could be the result of a streaming glitch."
                )
                continue
            except StopIteration:
                # Handle the end of the iterator
                self.reset()
                if not self.infinite:
                    logger.warning(f"Dataset {self.dataset_name} has run out of data")
                    break
                else:
                    logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                    iterator = self._get_data_iter()

            # Use the dataset-specific preprocessor
            sample_dict = self._data_processor(
                sample,
                self._t5_tokenizer,
                self._clip_tokenizer,
                output_size=self.job_config.training.img_size,
            )

            # skip low quality image or image with color channel = 1
            if "image" in sample_dict and sample_dict["image"] is None:
                logger.warning(
                    f"Low quality image {sample['__key__']} is skipped in Flux Dataloader"
                )
                continue

            # Classifier-free guidance: Replace some of the strings with empty strings.
            # Distinct random seed is initialized at the beginning of training for each FSDP rank.
            dropout_prob = self.job_config.training.classifer_free_guidance_prob
            if dropout_prob > 0.0 and random.random() < dropout_prob:
                if "t5_tokens" in sample_dict:
                    if self._t5_empty_token is None or self._clip_empty_token is None:
                        raise ValueError("Tokenizers must be provided.")
                    sample_dict["t5_tokens"] = self._t5_empty_token.clone()
                    sample_dict["clip_tokens"] = self._clip_empty_token.clone()
                elif "t5_encodings" in sample_dict:
                    sample_dict["drop_encodings"] = True
            else:
                if "t5_encodings" in sample_dict:
                    sample_dict["drop_encodings"] = False
            self._sample_idx += 1

            labels = (
                sample_dict.pop("image")
                if "image" in sample_dict
                else (sample_dict.pop("mean"), sample_dict.pop("logvar"))
            )

            yield sample_dict, labels

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict.get("sample_idx", 0)
        self._restored_checkpoint = True  # Mark that we've loaded from a checkpoint

    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
        }


def build_flux_train_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    tokenizer: FluxTokenizer | None,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    return _build_flux_dataloader(
        dataset_name=job_config.training.dataset,
        dataset_path=job_config.training.dataset_path,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        job_config=job_config,
        tokenizer=tokenizer,
        infinite=infinite,
        batch_size=job_config.training.batch_size,
        subset_size=job_config.training.subset_size,
    )


def build_flux_val_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    tokenizer: FluxTokenizer | None,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    if job_config.eval.dataset is None:
        return None
    return _build_flux_dataloader(
        dataset_name=job_config.eval.dataset,
        dataset_path=job_config.eval.dataset_path,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        job_config=job_config,
        tokenizer=tokenizer,
        infinite=infinite,
        batch_size=job_config.eval.batch_size,
    )


def _build_flux_dataloader(
    dataset_name: str,
    dataset_path: str,
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    # This parameter is not used, keep it for compatibility
    tokenizer: FluxTokenizer | None,
    infinite: bool = True,
    include_sample_id: bool = False,
    batch_size: int = 4,
    subset_size: int | None = None,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    if job_config.encoder.t5_encoder and job_config.encoder.clip_encoder:
        t5_tokenizer, clip_tokenizer = build_flux_tokenizer(job_config)
    else:
        t5_tokenizer, clip_tokenizer = None, None

    ds = FluxDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        t5_tokenizer=t5_tokenizer,
        clip_tokenizer=clip_tokenizer,
        job_config=job_config,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        include_sample_id=include_sample_id,
        subset_size=subset_size,
    )

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
