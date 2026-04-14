# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any, cast

import torch
import tyro
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
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

        # Variables for checkpointing
        self._sample_idx = 0
        self._epoch: int = 0
        self._inputs_buffer: list[int] = []
        self._positions_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def _normalize_positions(self, positions: list[int]) -> list[int]:
        offset = positions[0]
        if offset > 0:
            for i, p in enumerate(positions):
                if p == 0:
                    break
                positions[i] = p - offset
        return positions

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )

                self._inputs_buffer.extend(sample_tokens)
                # Per-document positions reset at document boundaries,
                # matching inference frameworks (e.g. vLLM) that start
                # positions at 0 per request.  Positions wrap at seq_len
                # to stay within the RoPE cache, effectively chunking
                # long documents into seq_len-sized segments.
                # TODO: make overflow policy configurable (chunk / truncate / drop).
                self._positions_buffer.extend(range(len(sample_tokens)))
                self._sample_idx += 1

                while len(self._inputs_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._inputs_buffer[:max_buffer_token_len])
                    pos = torch.LongTensor(
                        self._normalize_positions(
                            self._positions_buffer[:max_buffer_token_len]
                        )
                    )
                    # update buffers to the remaining tokens
                    self._inputs_buffer = self._inputs_buffer[max_buffer_token_len:]
                    self._positions_buffer = self._positions_buffer[
                        max_buffer_token_len:
                    ]

                    input = x[:-1]
                    label = x[1:]
                    positions = pos[:-1]
                    yield {"input": input, "positions": positions}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                self._epoch += 1
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._inputs_buffer = state_dict["inputs_buffer"]
        if "positions_buffer" not in state_dict:
            logger.warning(
                "Checkpoint missing 'positions_buffer'. Falling back to empty buffer. "
                "RoPE positions may be incorrect with block_causal attention."
            )
        self._positions_buffer = state_dict.get("positions_buffer", [])

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {
            "inputs_buffer": self._inputs_buffer,
            "positions_buffer": self._positions_buffer,
        }

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
            _state_dict["epoch"] = self._epoch
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


class HuggingFaceTextDataLoader(ParallelAwareDataloader):
    """Configurable text dataloader that wraps HuggingFaceTextDataset.

    This dataloader can be used for both training and validation by
    configuring the appropriate dataset, seq_len, batch_size, etc.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "c4_test"
        """Dataset to use"""

        infinite: bool = True
        """Whether to loop the dataset infinitely"""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        hf_ds = HuggingFaceTextDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }

        super().__init__(
            hf_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )


class ChatDataset(IterableDataset, Stateful):
    """Dataset for single-turn chat/instruction-tuning.

    Tokenizes [user, assistant] message pairs, masks prompt tokens with
    IGNORE_INDEX in labels, and uses greedy sequence packing with
    per-document positions. Implements Stateful for checkpointing.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: BaseTokenizer,
        sample_processor: Callable,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        if tokenizer.eos_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_id set. "
                "ChatDataset requires a tokenizer with a valid EOS token."
            )

        self._original_data = split_dataset_by_node(dataset, dp_rank, dp_world_size)
        self._data = self._original_data
        self._tokenizer = tokenizer
        self._eos_id = tokenizer.eos_id
        self.seq_len = seq_len
        self.infinite = infinite
        self._sample_processor = sample_processor

        self._dataset_id = f"{dataset.info.dataset_name}/{dataset.split}"

        # Variables for checkpointing
        self._sample_idx = 0
        self._epoch: int = 0
        self._inputs_buffer: list[int] = []
        self._labels_buffer: list[int] = []
        self._positions_buffer: list[int] = []

        self._logged_first_sample = False

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        """Validate that messages are a single-turn [user, assistant] pair."""
        # TODO: expand this to multi-turn
        if len(messages) != 2:
            raise ValueError(
                f"Expected single-turn [user, assistant], got {len(messages)} messages"
            )
        if messages[0]["role"] != "user":
            raise ValueError(
                f"First message must be 'user', got '{messages[0]['role']}'"
            )
        if messages[1]["role"] != "assistant":
            raise ValueError(
                f"Second message must be 'assistant', got '{messages[1]['role']}'"
            )

    def _tokenize_sample(
        self, sample: dict[str, Any]
    ) -> tuple[list[int], list[int]] | None:
        """Tokenize a single-turn sample and create input/label pairs.

        Returns (input_ids, label_ids) where input_ids = tokens[:-1] and
        label_ids = tokens[1:] with prompt tokens masked as IGNORE_INDEX.
        Returns None if the sample exceeds seq_len (dropped to avoid
        training on truncated responses).

        Uses incremental prefix re-tokenization to find the prompt/response
        token boundary, avoiding BPE merge errors.
        """
        messages = self._sample_processor(sample)
        self._validate_messages(messages)

        full_text = self._tokenizer.apply_chat_template(messages)
        # Strip extra newline and ensure the sequence ends with EOS without duplicates
        full_text = full_text.rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._eos_id:
            full_tokens.append(self._eos_id)

        if not self._logged_first_sample:
            logger.info(f"[ChatDataset] First sample full:\n{full_text}")
            self._logged_first_sample = True

        # Drop examples exceeding seq_len rather than truncating.
        if len(full_tokens) - 1 > self.seq_len:
            logger.debug(
                f"Dropping sample {self._sample_idx}: "
                f"tokens exceeds seq_len {self.seq_len}"
            )
            return None

        input_ids = full_tokens[:-1]
        label_ids = full_tokens[1:]

        # Find prompt/response boundary by tokenizing just the user message
        # with add_generation_prompt=True.
        prompt_text = self._tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        prompt_len = len(prompt_tokens)

        # Labels are shifted by one token, so the first assistant token is
        # predicted at index prompt_len - 1 and must remain unmasked.
        mask_end = min(max(prompt_len - 1, 0), len(label_ids))
        label_ids[:mask_end] = [IGNORE_INDEX] * mask_end

        return input_ids, label_ids

    def __iter__(self):
        yield from self._iter_greedy_packed()

    def _iter_greedy_packed(self):
        """Greedy packing: pack examples sequentially until seq_len is full.
        Document boundaries are marked by EOS tokens between packed examples.
        The model's flex/varlen attention mask uses these EOS positions to
        prevent cross-document attention.
        """
        while True:
            for sample in self._get_data_iter():
                # pyrefly: ignore [bad-argument-type]
                result = self._tokenize_sample(sample)
                if result is None:
                    self._sample_idx += 1
                    continue

                input_ids, label_ids = result
                remaining = self.seq_len - len(self._inputs_buffer)

                # If the example doesn't fit, pad and yield current buffer
                if len(input_ids) > remaining and len(self._inputs_buffer) > 0:
                    pad_len = remaining
                    self._inputs_buffer.extend([self._eos_id] * pad_len)
                    self._labels_buffer.extend([IGNORE_INDEX] * pad_len)
                    self._positions_buffer.extend(range(pad_len))

                    yield self._flush_buffers()

                # Add example to buffer with positions resetting to 0
                self._inputs_buffer.extend(input_ids)
                self._labels_buffer.extend(label_ids)
                self._positions_buffer.extend(range(len(input_ids)))
                self._sample_idx += 1

                if len(self._inputs_buffer) == self.seq_len:
                    yield self._flush_buffers()

            # Flush remaining buffer at end of data
            if len(self._inputs_buffer) > 0:
                pad_len = self.seq_len - len(self._inputs_buffer)
                if pad_len > 0:
                    self._inputs_buffer.extend([self._eos_id] * pad_len)
                    self._labels_buffer.extend([IGNORE_INDEX] * pad_len)
                    self._positions_buffer.extend(range(pad_len))

                yield self._flush_buffers()

            if not self.infinite:
                logger.warning(f"Chat dataset '{self._dataset_id}' has run out of data")
                break
            else:
                self._sample_idx = 0
                self._epoch += 1
                if isinstance(self._data, Dataset):
                    self._data = cast(
                        Dataset, self._original_data.shuffle(seed=42 + self._epoch)
                    )
                elif hasattr(self._data, "set_epoch"):
                    self._data.set_epoch(self._epoch)
                logger.warning(
                    f"Chat dataset '{self._dataset_id}' is being re-looped "
                    f"(epoch {self._epoch})"
                )

    def _flush_buffers(self):
        """Convert buffers to tensors, clear them, and return the batch."""
        input_tensor = torch.tensor(self._inputs_buffer, dtype=torch.long)
        label_tensor = torch.tensor(self._labels_buffer, dtype=torch.long)
        positions_tensor = torch.tensor(self._positions_buffer, dtype=torch.long)
        self._inputs_buffer = []
        self._labels_buffer = []
        self._positions_buffer = []
        return {"input": input_tensor, "positions": positions_tensor}, label_tensor

    def state_dict(self):
        _state_dict: dict[str, Any] = {
            "epoch": self._epoch,
            "inputs_buffer": self._inputs_buffer,
            "labels_buffer": self._labels_buffer,
            "positions_buffer": self._positions_buffer,
        }

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            _state_dict["data"] = self._data.state_dict()

        return _state_dict

    def load_state_dict(self, state_dict):
        self._epoch = state_dict["epoch"]
        self._inputs_buffer = state_dict["inputs_buffer"]
        self._labels_buffer = state_dict["labels_buffer"]
        self._positions_buffer = state_dict["positions_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
            # Replay shuffles so _data matches the order at checkpoint time
            if self._epoch > 0:
                self._data = cast(
                    Dataset, self._original_data.shuffle(seed=42 + self._epoch)
                )
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])


class ChatDataLoader(ParallelAwareDataloader):
    """Chat dataloader for instruction/conversation datasets."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_path: str | None = None
        """HuggingFace dataset path (e.g., 'openai/gsm8k') or local path. Required."""

        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
        """Extra kwargs passed to datasets.load_dataset()."""

        sample_processor: Annotated[Callable, tyro.conf.Suppress]
        """Callable(sample_dict) -> list[message_dict]. Set in config functions."""

        infinite: bool = True
        """Whether to loop the dataset infinitely. Might hang on multi-GPU."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        if not config.dataset_path:
            raise ValueError(
                "ChatDataLoader requires dataset_path to be set "
                "(e.g., 'openai/gsm8k' or 'json')."
            )

        dataset = load_dataset(config.dataset_path, **config.load_dataset_kwargs)

        chat_ds = ChatDataset(
            dataset=dataset,  # pyrefly: ignore [bad-argument-type]
            tokenizer=tokenizer,
            sample_processor=config.sample_processor,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }

        super().__init__(
            chat_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
