# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass
from logging import raiseExceptions
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from einops import rearrange, repeat
from PIL import Image
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.tokenizer.HFEmbedder import HFEmbedder
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchvision import transforms


def _process_cc12m_image(
    sample: dict[str, Any], output_size: int = 256
) -> tuple[str, torch.Tensor]:
    """
    Process CC12M dataset sample image.
    Steps:
        1. Ignore all the images smaller than output_size * output_size.
        2. Resize and crop the image to output_size * output_size.
    """
    img = sample[
        "jpg"
    ]  # NOTE: type is PIL Image, https://huggingface.co/docs/datasets/en/image_load

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
class MultiModalDatasetConfig:
    path: str
    loader: Callable
    data_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
DATASETS = {
    "cc12m": MultiModalDatasetConfig(
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


def _get_noise_latent(
    num_samples: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,  # NOTE(jianiw): From the paper, d=16, h = H/8, w = W/8
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator().manual_seed(seed),
    )


class FLUXDataLoader(StatefulDataLoader, BaseDataLoader):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        embedder: List[HFEmbedder],  # Tokenizer for text
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        batch_size: int = 1,
        seed: int = 0,
        device: str = "cpu",
        job_config: JobConfig = None,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.seed = seed
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        assert len(embedder) == 2

        self._t5_embedder = embedder[0]
        self._clip_embedder = embedder[1]
        self._data_processor = data_processor
        self._image_transformer = transforms.ToTensor()

        self.infinite = infinite
        self.batch_size = batch_size
        self.device = device

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_target_imgs: list[torch.Tensor] = []
        self._all_prompts: list[str] = []
        self.job_config = job_config

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def _prepare(self):
        target_img = torch.stack(self._all_target_imgs).to(self.device)

        sample_t5_embedding = self._t5_embedder(self._all_prompts)
        txt_ids = torch.zeros(self.batch_size, sample_t5_embedding.shape[1], 3)
        sample_clip_embedding = self._clip_embedder(self._all_prompts)

        noise_latent = _get_noise_latent(
            self.batch_size,
            target_img.shape[0],
            target_img.shape[1],
            dtype=torch.bfloat16,
            seed=self.seed,
        )

        _, _, h, w = noise_latent.shape  # (16, 256/8, 256/8) = (16, 32, 32)
        # patchify the noise latent, p = 2
        noise_latent = rearrange(
            noise_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
        )

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=self.batch_size)

        output = {
            "img": noise_latent.to(self.device),
            "img_ids": img_ids.to(self.device),
            "txt": sample_t5_embedding.to(self.device),
            "txt_ids": txt_ids.to(self.device),
            "vec": sample_clip_embedding.to(self.device),
            "target": target_img,
        }

        return output

    def __iter__(self):
        samples_cnt = 0

        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1

                # Use the dataset-specific text processor
                sample_prompt, sample_image = self._data_processor(sample)
                if sample_prompt is None or sample_image is None:
                    continue

                samples_cnt += 1

                self._all_target_imgs.extend(self._image_transformer(sample_image))
                self._all_prompts.extend(sample_prompt)

                if samples_cnt == self.batch_size:
                    batched_data = self._prepare()
                    samples_cnt = 0
                    self._all_prompts = []
                    self._all_target_imgs = []
                    yield batched_data

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


def build_flux_dataloader(
    dp_world_size: int,
    dp_rank: int,
    embedder: List[HFEmbedder],
    job_config: JobConfig,
    infinite: bool = True,
    device: str = "cpu",
) -> FLUXDataLoader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seed = job_config.training.seed

    return FLUXDataLoader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        embedder=embedder,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        batch_size=batch_size,
        seed=seed,
        device=device,
        job_config=job_config,
    )
