# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
from typing import Any

import numpy as np
import PIL
import torch
from datasets import Dataset, load_dataset, load_from_disk

from torchtitan.experiments.flux.dataset.flux_dataset import (
    _process_cc12m_image,
    DATASETS,
    TextToImageDatasetConfig,
)
from torchtitan.experiments.flux.dataset.tokenizer import FluxTokenizer

# Avoid PIL.Image.DecompressionBombError
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def _coco_data_processor(
    sample: dict[str, Any],
    t5_tokenizer: FluxTokenizer,
    clip_tokenizer: FluxTokenizer,
    output_size: int = 256,
) -> dict[str, Any]:
    """
    Preprocess COCO dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_encoder: T5 encoder
        clip_encoder: CLIP encoder
        output_size: The output image size

    """

    img_key = "png" if "png" in sample else "jpg"
    img = _process_cc12m_image(
        sample[img_key], output_size=output_size, skip_low_resolution=False
    )
    t5_tokens = t5_tokenizer.encode(sample["txt"])
    clip_tokens = clip_tokenizer.encode(sample["txt"])

    # Extract id from JSON metadata if available
    sample_id = None

    metadata = sample["json"]  # Already parsed as dict by webdataset
    if "id" in metadata:
        sample_id = metadata["id"]
    timestep = metadata["timestep"]
    result = {
        "image": img,
        "clip_tokens": clip_tokens,  # type: List[int]
        "t5_tokens": t5_tokens,  # type: List[int]
        "txt": sample["txt"],
        "timestep": timestep,
    }

    # Add id if available
    if sample_id is not None:
        result["id"] = sample_id

    return result


DATASETS["coco"] = TextToImageDatasetConfig(
    path="/dataset/coco",
    loader=lambda path: load_dataset(
        "webdataset",
        split="train",
        data_dir=path,
        streaming=True,
    ),
    data_processor=_coco_data_processor,
)


def deserialize_numpy_array(data: bytes) -> torch.Tensor:
    """Deserialize numpy array from bytes."""
    buffer = io.BytesIO(data)
    uint16_data = np.load(buffer)
    return torch.from_numpy(uint16_data).view(torch.bfloat16)


def _coco_data_processor_from_encodings(
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
    sample["timestep"] = sample["timestep"]
    sample_id = sample.pop("__key__")
    if include_sample_id:
        sample["id"] = sample_id

    return sample


DATASETS["coco_preprocessed"] = TextToImageDatasetConfig(
    path="/dataset/coco_preprocessed",
    loader=lambda path: load_from_disk(path),
    data_processor=_coco_data_processor_from_encodings,
)


def create_dummy_dataset(num_samples: int = 10000):
    """Create a dummy Hugging Face Dataset for testing."""
    # Create dummy images as PIL Images to ensure compatibility

    dummy_image = PIL.Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )
    timesteps = np.arange(num_samples) % 8
    data = {
        "txt": ["A photo of a cat"] * num_samples,
        "png": [dummy_image] * num_samples,
        "json": [{"timestep": timestep} for timestep in timesteps],
    }
    return Dataset.from_dict(data)


DATASETS["dummy"] = TextToImageDatasetConfig(
    path="dummy",
    loader=lambda path: create_dummy_dataset(128),
    data_processor=_coco_data_processor,
)
