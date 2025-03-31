# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from mm_collator import MultiModalCollator
from tokenizer.tiktoken import IGNORE_INDEX, Tokenizer
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from transform import CLIPTransform
from utils import load_image

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


def _load_obelics_dataset(dataset_path: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, split="train", streaming=True)


def _process_obelics_sample(
    sample: dict[str, Any], image_token: str = "<|image|>"
) -> Dict[str, List[Union[str, "PIL.Image.Image"]]]:
    """
    This function formats samples from the OBELICS dataset
    Returns:
        Dict[str, Any]: The transformed sample with the following fields:
            - images: List[PIL.Image.Image] with the loaded images
            - text: str with the text of the sample ready to be tokenized including the image tokens
    Example:
        >>> formatted_sample = format_obelics(sample, image_token="<|image|>")
        >>> print(formatted_sample["text"])
        ... "<|image|><|image|><|image|> The elephant look cute!<|image|><|image|> The cats are sad :("
    """
    sample_images = [image for image in sample["images"] if image is not None]
    sample_text = [
        text if text is not None else image_token for text in sample["texts"]
    ]
    return {
        "images": [load_image(image) for image in sample_images],
        "text": "".join(map(str, sample_text)),
    }


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
MM_DATASETS = {
    "obelics": DatasetConfig(
        path="HuggingFaceM4/OBELICS",
        loader=_load_obelics_dataset,
        sample_processor=_process_obelics_sample,
    ),
}


def _validate_mm_dataset(
    dataset_name: str, dataset_path: str = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in MM_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(MM_DATASETS.keys())}"
        )

    config = MM_DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class MultiModalDataset(IterableDataset, Stateful):
    """PyTorch MultiModal Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently ONLY support the OBELICS dataset

    Example use:
    >>> ds = MultiModalDataset(dataset_name="OBELICS", tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        image_token: str = "<|image|>",
        tile_size: int = 448,
        max_num_tiles: int = 4,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, sample_processor = _validate_mm_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        # TODO: support shuffling
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._sample_processor = sample_processor
        self.image_token = (
            image_token  # TODO(tj.solergibert) Add `image_token` to JobConfig
        )
        # TODO(tj.solergibert) Add `tile_size` & `max_num_tiles` to JobConfig
        self.transform_image = CLIPTransform(
            image_mean=(
                0.48145466,
                0.4578275,
                0.40821073,
            ),  # TODO(tj.solergibert) What should we do with `image_mean` & `image_std`?,
            image_std=(0.26862954, 0.26130258, 0.27577711),
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
        )

        # variables for checkpointing
        self._sample_idx = 0

    def __iter__(self):

        while True:
            for sample in self._get_data_iter():
                try:
                    sample = self._sample_processor(
                        sample, image_token=self.image_token
                    )
                except Exception:
                    continue
                self._sample_idx += 1

                # CLIP Transform
                encoder_input = {"images": [], "aspect_ratio": []}
                for image in sample["images"]:
                    out = self.transform_image(image)
                    encoder_input["images"].append(out["image"])
                    encoder_input["aspect_ratio"].append(out["aspect_ratio"])
                sample["encoder_input"] = encoder_input

                # Tokenize
                tokens = self._tokenizer.encode(
                    sample["text"],
                    bos=True,
                    eos=True,
                    allowed_special=set(["<|image|>"]),
                )
                sample["input_ids"] = torch.LongTensor(tokens[:-1])
                sample["labels"] = torch.LongTensor(tokens[1:])
                # Mask BOS, EOS & image tokens from the loss
                sample["labels"] = torch.where(
                    torch.isin(
                        sample["labels"],
                        torch.LongTensor(
                            [
                                self._tokenizer.bos_id,
                                self._tokenizer.eos_id,
                                self._tokenizer.image_id,
                            ]
                        ),
                    ),
                    IGNORE_INDEX,
                    sample["labels"],
                )
                # Truncate
                sample["input_ids"], sample["labels"] = (
                    sample["input_ids"][: self.seq_len],
                    sample["labels"][: self.seq_len],
                )
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}


def build_mm_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len
    pad_max_tiles = 4  # TODO(tj.solergibert) Add `pad_max_tiles` to JobConfig
    padding_idx = 128004  # TODO(tj.solergibert) Add `padding_idx` to JobConfig

    hf_ds = MultiModalDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    collate_fn = MultiModalCollator(
        padding_idx=padding_idx, pad_max_tiles=pad_max_tiles
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
