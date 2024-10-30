# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.datasets import DPAwareDataLoader
from torchtitan.datasets.multimodal import (
    format_obelics,
    Llama3VisionTransform,
    MultiModalCollator,
)
from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

# map from dataset name to a dataset repository on the HF hub
_supported_datasets = {
    "OBELICS": "HuggingFaceM4/OBELICS",
}


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
        tokenizer: Tokenizer,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        # Do NOT allow user to pass any other dataset which is not OBELICS
        if dataset_name not in _supported_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not supported. "
                f"Supported datasets are: {list(_supported_datasets.keys())}"
            )

        dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")
        ds = load_dataset(dataset_path, split="train", streaming=True)

        # TODO: support shuffling
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.infinite = infinite
        self.image_token = "<|image|>"  # TODO(tj.solergibert) Hardcoded!

        self.format = format_obelics
        self.transform = Llama3VisionTransform(
            tokenizer=tokenizer,
            tile_size=448,
            patch_size=14,
            max_num_tiles=4,
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
        )
        # NOTE(tj.solergibert) 560 for Instruct models, 448 for pretrain
        # https://github.com/pytorch/torchtune/blob/0cc1b1f6a2a9c54ca640c4eb0a4d0b94ba94bb04/torchtune/models/llama3_2_vision/_model_builders.py#L92
        # https://huggingface.co/meta-llama/Llama-3.2-11B-Vision/blob/3f2e93603aaa5dd142f27d34b06dfa2b6e97b8be/preprocessor_config.json#L22

        # variables for checkpointing
        self._sample_idx = 0

    def __iter__(self):

        while True:
            for sample in self._get_data_iter():
                # Format sample into `Llama3VisionTransform` format
                try:
                    processed_sample = self.format(sample, image_token=self.image_token)
                except Exception:
                    continue
                assert len(processed_sample["images"]) == processed_sample[
                    "text"
                ].count(self.image_token)
                self._sample_idx += 1
                # Transform sample
                processed_sample = self.transform(processed_sample)
                yield processed_sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        # As skipping to the end throws an error in case of map-style dataset, return an empty iterator
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}


def build_mm_data_loader(
    dataset_name: str,
    tokenizer: Tokenizer,
    batch_size: int,
    world_size,
    rank,
    infinite: bool = True,
):
    mm_ds = MultiModalDataset(dataset_name, tokenizer, world_size, rank, infinite)

    collator = MultiModalCollator(padding_idx=0, pad_max_tiles=4)

    return DPAwareDataLoader(rank, mm_ds, batch_size=batch_size, collator_fn=collator)
