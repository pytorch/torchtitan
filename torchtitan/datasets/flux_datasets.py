# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from einops import rearrange, repeat
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.config_manager import JobConfig
from torchtitan.datasets.tokenizer.HFEmbedder import HFEmbedder
from torchtitan.tools.logging import logger


def _process_cc12m_image(
    sample: dict[str, Any], output_size: int = 256
) -> torch.Tensor:
    """
    Process CC12M dataset sample image.
    Steps:
        1. Ignore all the images smaller than output_size * output_size.
        2. Resize and crop the image to output_size * output_size.
    """
    img = sample[
        "jpg"
    ]  # NOTE: type should be PIL Image, https://huggingface.co/docs/datasets/en/image_load
    # TODO(jianiw): add image open, processing logic here

    # TODO(jianiw): Check the output size, (3, 256, 256) ??
    assert img.shape[0] == img.shape[1] == output_size
    return torch.Tensor()


@dataclass
class MultiModalDatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable
    image_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
DATASETS = {
    "cc12m": MultiModalDatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train"),
        text_processor=lambda sample: sample["text"],
        image_processor=_process_cc12m_image,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: Optional[str] = None
) -> tuple[str, Callable, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor, config.image_processor


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


class FLUXDataLoader(StatefulDataLoader, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        embedder: List[HFEmbedder],  # Tokenizer for text
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        batch_size: int = 1,
        seed: int = 0,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor, image_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        # TODO(jianiw): Switch to use toml config to set the seed
        self.seed = seed
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        assert len(embedder) == 2

        self._t5_embedder = embedder[0]
        self._clip_embedder = embedder[1]
        self._text_processor = text_processor
        self._image_processor = image_processor  # TODO(jianiw)

        self.seq_len = seq_len
        self.infinite = infinite
        self.batch_size = batch_size

        # NOTE(jianiw): No need at this moment
        # # Variables for checkpointing
        self._sample_idx = 0
        # self._all_tokens: list[int] = []
        self._all_target_imgs: list[torch.Tensor] = []
        self._all_prompts: list[str] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def _prepare(self):
        target_img = torch.stack(self._all_target_imgs)

        sample_t5_embedding = self._t5_embedder(self._all_prompts)
        txt_ids = torch.zeros(self.batch_size, sample_t5_embedding.shape[1], 3)
        sample_clip_embedding = self._clip_tokenizer(self._all_prompts)

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

        return {
            "img": noise_latent,
            "img_ids": img_ids,
            "txt": sample_t5_embedding,
            "txt_ids": txt_ids,
            "vec": sample_clip_embedding,
            "target": target_img,
        }

    def __iter__(self):
        samples_cnt = 0

        while True:
            for sample in self._get_data_iter():
                samples_cnt += 1

                # Use the dataset-specific text processor
                sample_promt = self._text_processor(sample)
                sample_image = self._image_processor(sample)

                self._all_target_imgs.extend(sample_image)
                self._all_prompts.extend(sample_promt)
                self._sample_idx += 1

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
        # TODO(jianiw): Check if we need this function
        pass

    def state_dict(self):
        pass


def build_flux_dataloader(
    dp_world_size: int,
    dp_rank: int,
    embedder: List[HFEmbedder],
    job_config: JobConfig,
    infinite: bool = True,
) -> FLUXDataLoader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len
    seed = job_config.training.seed

    return FLUXDataLoader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        embedder=embedder,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        batch_size=batch_size,
        seed=seed,
    )
