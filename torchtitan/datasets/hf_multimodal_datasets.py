# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from einops import rearrange
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.utils import add_padding
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
    height: int,
    width: int,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        16,  # NOTE(jianiw): From the paper, d=16, h = H/8, w = W/8
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator().manual_seed(seed),
    )


class HuggingFaceMultiModalDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: List[Tokenizer],  # Tokenizer for text
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor, image_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        # TODO(jianiw): Switch to use toml config to set the seed
        self.seed = 0
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        # TODO(jianiw): Find aa beter way to handle multiple tokenizers
        assert len(tokenizer) == 2

        self._t5_tokenizer = tokenizer[0]
        self._clip_tokenizer = tokenizer[1]

        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        self._image_processor = image_processor  # TODO(jianiw)

        # NOTE(jianiw): No need at this moment
        # # Variables for checkpointing
        # self._sample_idx = 0
        # self._all_tokens: list[int] = []

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
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_image = self._image_processor(sample)

                # TODO(jianiw): Implement encoder, eg: https://github.com/black-forest-labs/flux/blob/716724eb276d94397be99710a0a54d352664e23b/src/flux/modules/conditioner.py
                sample_t5_tokens = self._t5_tokenizer.encode(
                    sample_text,
                    padding="max_length",
                    max_length=self.seq_len,
                    return_overflowing_tokens=False,  # Do not return overflowing tokens
                    return_length=False,
                    return_tensors="pt",
                )

                # TODO: check the encoder output
                txt_ids = torch.zeros(sample_t5_tokens.shape[1], 3)

                # TODO(jianiw): We don't need to pad the tokens here, since the tokenizer  do it for us
                # sample_t5_tokens = add_padding(
                #     sample_t5_tokens, padding_item=0, target_len=self.seq_len
                # )

                sample_clip_tokens = self._clip_tokenizer.encode(
                    sample_text,
                    padding="max_length",
                    max_length=self.seq_len,
                    return_overflowing_tokens=False,  # Do not return overflowing tokens
                    return_length=False,
                    return_tensors="pt",
                )

                noise_latent = _get_noise_latent(
                    sample_image.shape[0],  # TODO(jianiw): Check the noise latent shape
                    sample_image.shape[1],
                    dtype=torch.bfloat16,
                    seed=self.seed,
                )

                _, h, w = noise_latent.shape  # (16, 256/8, 256/8) = (16, 32, 32)
                # patchify the noise latent, p = 2
                noise_latent = rearrange(
                    noise_latent, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2
                )

                # TODO(jianiw): Check what does this part of code do
                img_id = torch.zeros(h // 2, w // 2, 3)
                img_id[..., 1] = img_id[..., 1] + torch.arange(h // 2)[:, None]
                img_id[..., 2] = img_id[..., 2] + torch.arange(w // 2)[None, :]
                img_id = rearrange(img_id, "h w c -> (h w) c")

                yield {
                    "img": noise_latent,
                    "img_id": img_id,
                    "txt": sample_t5_tokens,
                    "txt_id": txt_id,
                    "vec": sample_clip_tokens,
                    "target": sample_image,
                }

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


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: List[Tokenizer],
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
