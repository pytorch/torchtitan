# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import torch
import tyro
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


class SFTDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: BaseTokenizer,
        sample_processor: Callable,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        pack_sequences: bool = True,
    ) -> None:
        if tokenizer.eos_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_id set. "
                "SFT requires a tokenizer with a valid EOS token for padding."
            )

        self._data = split_dataset_by_node(dataset, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self._eos_id = tokenizer.eos_id
        self.seq_len = seq_len
        self.infinite = infinite
        self.pack_sequences = pack_sequences
        self._sample_processor = sample_processor
        self._dataset_id = f"{dataset.info.dataset_name}/{dataset.split}"

        # Variables for checkpointing
        self._sample_idx = 0
        self._epoch: int = 0
        self._pack_buffer_input: list[int] = []
        self._pack_buffer_label: list[int] = []

        self._logged_first_sample = False

    def _get_data_iter(self):
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def _tokenize_sample(
        self, sample: dict[str, Any]
    ) -> tuple[list[int], list[int]] | None:
        """Tokenize a sample and create input/label pairs with label masking.

        Returns (input_ids, label_ids) where input_ids = tokens[:-1] and
        label_ids = tokens[1:] with non-assistant tokens masked as IGNORE_INDEX.
        Returns None if the sample is too short (< 2 tokens) or exceeds
        seq_len (dropped to avoid training on truncated responses).

        Uses incremental prefix re-tokenization (delta approach) to determine
        per-message token boundaries, avoiding BPE merge errors at the
        prompt/response boundary.
        """
        messages = self._sample_processor(sample)
        full_text = self._tokenizer.apply_chat_template(messages)
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=True)

        # Log only the first sample for debugging
        if not self._logged_first_sample:
            logger.info(f"[SFT] First sample full:\n{full_text}")
            self._logged_first_sample = True

        # Need at least 2 tokens for next-token prediction (input[:-1], label[1:])
        if len(full_tokens) < 2:
            logger.debug(f"Dropping sample {self._sample_idx}: too short to form input/label pair \n{full_text}")
            return None

        # Drop examples exceeding seq_len rather than truncating. Truncation
        # cuts the end of the response (often the final answer), which teaches
        # the model to produce incomplete outputs. Axolotl defaults to drop
        # for the same reason. If many examples are dropped, increase seq_len.
        if len(full_tokens) - 1 > self.seq_len:
            logger.debug(
                f"Dropping sample {self._sample_idx}: tokens exceeds seq_len {self.seq_len} \n {full_text}"
            )
            return None

        # Next-token prediction shift: input = tokens[:-1], labels = tokens[1:]
        # The trainer and loss function do NOT shift again.
        input_ids = full_tokens[:-1]
        label_ids = list(full_tokens[1:])

        # Build per-token mask using incremental prefix re-tokenization.
        # For each message, tokenize the conversation up to that point.
        # The delta (new tokens) belongs to that message. Mask non-assistant
        # deltas so we only train on assistant responses.
        prev_token_len = 0
        for i, message in enumerate(messages):
            prefix_messages = messages[: i + 1]
            is_last = i == len(messages) - 1
            prefix_text = self._tokenizer.apply_chat_template(
                prefix_messages,
                add_generation_prompt=not is_last,
            )
            prefix_tokens = self._tokenizer.encode(
                prefix_text, add_bos=True, add_eos=False
            )
            curr_token_len = len(prefix_tokens)

            if message["role"] != "assistant":
                # Mask this message's tokens in labels (accounting for shift)
                mask_start = max(prev_token_len - 1, 0)
                mask_end = min(curr_token_len - 1, len(label_ids))
                for j in range(mask_start, mask_end):
                    label_ids[j] = IGNORE_INDEX

            prev_token_len = curr_token_len

        return input_ids, label_ids

    def __iter__(self):
        if self.pack_sequences:
            yield from self._iter_greedy_packed()
        else:
            yield from self._iter_unpacked()

    def _iter_unpacked(self):
        """Yield each example independently, padded to seq_len."""
        while True:
            for sample in self._get_data_iter():
                result = self._tokenize_sample(sample)
                self._sample_idx += 1
                if result is None:
                    continue

                input_ids, label_ids = result

                # Pad to seq_len
                pad_len = self.seq_len - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [self._eos_id] * pad_len
                    label_ids = label_ids + [IGNORE_INDEX] * pad_len

                input_tensor = torch.tensor(input_ids, dtype=torch.long)
                label_tensor = torch.tensor(label_ids, dtype=torch.long)
                yield {"input": input_tensor}, label_tensor

            if not self.infinite:
                logger.warning(f"SFT dataset '{self._dataset_id}' has run out of data")
                break
            else:
                self._sample_idx = 0
                self._epoch += 1
                self._data = self._data.shuffle(seed=42 + self._epoch)
                logger.warning(f"SFT dataset '{self._dataset_id}' is being re-looped (epoch {self._epoch})")

    def _iter_greedy_packed(self):
        """Greedy packing: pack examples sequentially until seq_len is full.

        Each example is added to the current sequence buffer in order. When the
        next example doesn't fit in the remaining space, the buffer is padded and
        yielded. This is simple and fast but may leave padding gaps when examples
        don't divide evenly into seq_len.

        Document boundaries are marked by EOS tokens between packed examples.
        The model's flex/varlen attention mask uses these EOS positions to
        prevent cross-document attention (see get_document_mask_mod).
        """
        while True:
            for sample in self._get_data_iter():
                result = self._tokenize_sample(sample)
                self._sample_idx += 1
                if result is None:
                    continue

                input_ids, label_ids = result
                remaining = self.seq_len - len(self._pack_buffer_input)

                # If the example doesn't fit, pad and yield current buffer
                if len(input_ids) > remaining and len(self._pack_buffer_input) > 0:
                    pad_len = remaining
                    self._pack_buffer_input.extend([self._eos_id] * pad_len)
                    self._pack_buffer_label.extend([IGNORE_INDEX] * pad_len)

                    input_tensor = torch.tensor(
                        self._pack_buffer_input, dtype=torch.long
                    )
                    label_tensor = torch.tensor(
                        self._pack_buffer_label, dtype=torch.long
                    )
                    self._pack_buffer_input = []
                    self._pack_buffer_label = []
                    yield {"input": input_tensor}, label_tensor

                # Add example to buffer
                self._pack_buffer_input.extend(input_ids)
                self._pack_buffer_label.extend(label_ids)

                # If buffer is exactly full, yield it. Buffer can never exceed
                # seq_len because all examples are <= seq_len (over-length
                # examples are dropped in _tokenize_sample).
                if len(self._pack_buffer_input) == self.seq_len:
                    input_tensor = torch.tensor(
                        self._pack_buffer_input, dtype=torch.long
                    )
                    label_tensor = torch.tensor(
                        self._pack_buffer_label, dtype=torch.long
                    )
                    self._pack_buffer_input = []
                    self._pack_buffer_label = []
                    yield {"input": input_tensor}, label_tensor

            # Flush remaining buffer at end of data
            if len(self._pack_buffer_input) > 0:
                pad_len = self.seq_len - len(self._pack_buffer_input)
                if pad_len > 0:
                    self._pack_buffer_input.extend([self._eos_id] * pad_len)
                    self._pack_buffer_label.extend([IGNORE_INDEX] * pad_len)

                input_tensor = torch.tensor(self._pack_buffer_input, dtype=torch.long)
                label_tensor = torch.tensor(self._pack_buffer_label, dtype=torch.long)
                self._pack_buffer_input = []
                self._pack_buffer_label = []
                yield {"input": input_tensor}, label_tensor

            if not self.infinite:
                logger.warning(f"SFT dataset '{self._dataset_id}' has run out of data")
                break
            else:
                self._sample_idx = 0
                self._epoch += 1
                self._data = self._data.shuffle(seed=42 + self._epoch)
                logger.warning(f"SFT dataset '{self._dataset_id}' is being re-looped (epoch {self._epoch})")

    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
            "epoch": self._epoch,
            "pack_buffer_input": self._pack_buffer_input,
            "pack_buffer_label": self._pack_buffer_label,
        }

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._epoch = state_dict.get("epoch", 0)
        self._pack_buffer_input = state_dict["pack_buffer_input"]
        self._pack_buffer_label = state_dict["pack_buffer_label"]


class SFTDataLoader(ParallelAwareDataloader):
    """SFT dataloader that wraps SFTDataset."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_path: str = ""
        """HuggingFace dataset path (e.g., 'openai/gsm8k') or local path."""

        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
        """Extra kwargs passed to datasets.load_dataset() (e.g., name, split, data_files)."""

        sample_processor: Annotated[Callable, tyro.conf.Suppress]
        """Callable(sample_dict) -> list[message_dict]. Set in config functions."""

        infinite: bool = True
        """Whether to loop the dataset infinitely."""

        pack_sequences: bool = True
        """Whether to pack multiple examples into a single sequence."""

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
        dataset = load_dataset(config.dataset_path, **config.load_dataset_kwargs)

        sft_ds = SFTDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            sample_processor=config.sample_processor,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            pack_sequences=config.pack_sequences,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }

        super().__init__(
            sft_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
