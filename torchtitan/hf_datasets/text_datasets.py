# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict
from functools import partial
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


# Harmony persona dataset loader
def _load_harmony_jsonl(dataset_path: str):
    """Load Harmony persona dataset from JSONL file."""
    import json

    def generate_samples():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                yield record

    # Convert generator to HF Dataset
    from datasets import Dataset as HFDataset

    samples = list(generate_samples())
    return HFDataset.from_list(samples)


def _process_harmony_text(sample: dict[str, Any]) -> str:
    """Process Harmony dataset sample - extract pre-formatted text."""
    return sample["text"]


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
    # JSONL datasets can be added here using _load_harmony_jsonl loader
    # Example:
    # "my_dataset": DatasetConfig(
    #     path="/path/to/dataset.jsonl",
    #     loader=_load_harmony_jsonl,
    #     sample_processor=_process_harmony_text,
    # ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
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
    return path, config.loader, config.sample_processor


class HuggingFacePackedDataset(IterableDataset, Stateful):
    """Dataset that packs complete samples without splitting across sequence boundaries.

    Unlike HuggingFaceTextDataset which concatenates and chunks at fixed intervals
    (potentially splitting samples), this class:
    1. Greedily packs complete samples until seq_len is reached
    2. Pads the remainder with pad_token
    3. Sets labels to -100 (ignore_index) for padding positions

    This ensures each sample is trained as a complete unit, which is important
    for fine-tuning tasks where sample boundaries matter.
    """

    IGNORE_INDEX = -100  # PyTorch cross_entropy default ignore_index

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        add_bos_eos: bool = True,
    ) -> None:
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        self.add_bos_eos = add_bos_eos

        # Get pad token - try pad_id, then look up <|endoftext|>, finally fall back to eos_id
        self.pad_id = getattr(tokenizer, "pad_id", None)
        if self.pad_id is None:
            # Try to get pad token from tokenizer's underlying tokenizer
            try:
                self.pad_id = tokenizer.tokenizer.token_to_id("<|endoftext|>")
            except (AttributeError, KeyError):
                # Tokenizer doesn't support token_to_id or token not in vocab
                logger.debug("Could not find <|endoftext|> token, will use eos_id fallback")
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_id
            logger.warning(
                f"No pad token found, using eos_id={self.pad_id} for padding"
            )

        # Variables for checkpointing
        self._sample_idx = 0
        self._pending_samples: list[
            list[int]
        ] = []  # Tokenized samples waiting to be packed

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))
        return iter(self._data)

    def __iter__(self):
        max_seq_len = self.seq_len + 1  # +1 for input/label shift

        while True:
            for sample in self._get_data_iter():
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=self.add_bos_eos, add_eos=self.add_bos_eos
                )
                self._sample_idx += 1

                # Skip samples that are too long (can't fit in a single sequence)
                if len(sample_tokens) > max_seq_len:
                    logger.warning(
                        f"Skipping sample {self._sample_idx} with {len(sample_tokens)} tokens "
                        f"(exceeds max {max_seq_len})"
                    )
                    continue

                self._pending_samples.append(sample_tokens)

                # Check if we can yield a packed sequence
                # We yield when the next sample won't fit
                while len(self._pending_samples) >= 2:
                    # Calculate how many samples fit
                    packed_tokens = []
                    samples_to_use = 0

                    for tokens in self._pending_samples:
                        if len(packed_tokens) + len(tokens) <= max_seq_len:
                            packed_tokens.extend(tokens)
                            samples_to_use += 1
                        else:
                            break

                    # Only yield if we have leftover samples (meaning we filled up)
                    if samples_to_use < len(self._pending_samples):
                        # Remove used samples
                        self._pending_samples = self._pending_samples[samples_to_use:]

                        # Pad and yield
                        num_real_tokens = len(packed_tokens)
                        num_padding = max_seq_len - num_real_tokens
                        packed_tokens.extend([self.pad_id] * num_padding)

                        x = torch.LongTensor(packed_tokens)
                        input_ids = x[:-1]
                        labels = x[1:].clone()

                        if num_padding > 1:
                            # Mask labels where INPUT is padding (num_padding - 1 positions)
                            # We keep one: predicting first PAD from last real token
                            labels[-(num_padding - 1):] = self.IGNORE_INDEX

                        yield {"input": input_ids}, labels
                    else:
                        # All pending samples fit, wait for more
                        break

            # End of data - flush any remaining samples
            if self._pending_samples:
                # Combine all pending into one final sequence
                packed_tokens = []
                for tokens in self._pending_samples:
                    if len(packed_tokens) + len(tokens) <= max_seq_len:
                        packed_tokens.extend(tokens)
                self._pending_samples = []

                if packed_tokens:
                    num_real_tokens = len(packed_tokens)
                    num_padding = max_seq_len - num_real_tokens
                    packed_tokens.extend([self.pad_id] * num_padding)

                    x = torch.LongTensor(packed_tokens)
                    input_ids = x[:-1]
                    labels = x[1:].clone()

                    if num_padding > 1:
                        # Mask labels where INPUT is padding (num_padding - 1 positions)
                        labels[-(num_padding - 1):] = self.IGNORE_INDEX

                    yield {"input": input_ids}, labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                self._sample_idx = 0
                self._pending_samples = []
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._pending_samples = state_dict.get("pending_samples", [])
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"pending_samples": self._pending_samples}
        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            _state_dict["data"] = self._data.state_dict()
        return _state_dict


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        add_bos_eos: bool = True,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        self.add_bos_eos = add_bos_eos

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=self.add_bos_eos, add_eos=self.add_bos_eos
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets.

    Uses HuggingFacePackedDataset if pack_samples=True (preserves sample boundaries),
    otherwise uses HuggingFaceTextDataset (concatenates and chunks).

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    pack_samples = getattr(job_config.training, "pack_samples", False)
    add_bos_eos = getattr(job_config.training, "add_bos_eos", True)

    if pack_samples:
        logger.info("Using packed dataset (preserves sample boundaries)")
        if not add_bos_eos:
            logger.info("BOS/EOS tokens disabled (using pre-formatted data)")
        hf_ds = HuggingFacePackedDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            add_bos_eos=add_bos_eos,
        )
    else:
        hf_ds = HuggingFaceTextDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            add_bos_eos=add_bos_eos,
        )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    dataloader_kwargs = {
        **asdict(job_config.validation.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
