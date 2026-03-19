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
        self._pack_buffer_positions: list[int] = []

        self._logged_first_sample = False

    def _get_data_iter(self):
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        """Validate that messages are a single-turn [user, assistant] pair."""
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

        Expects messages = [user_message, assistant_message] from the
        sample_processor. Returns (input_ids, label_ids) where
        input_ids = tokens[:-1] and label_ids = tokens[1:] with prompt
        tokens masked as IGNORE_INDEX. Returns None if the sample exceeds
        seq_len (dropped to avoid training on truncated responses).

        Uses incremental prefix re-tokenization to find the prompt/response
        token boundary, avoiding BPE merge errors. This is the same approach
        used by torchtune's HFTokenizer.tokenize_messages:
        https://github.com/pytorch/torchtune/blob/main/torchtune/modules/transforms/tokenizers/_hf_tokenizer.py
        """
        messages = self._sample_processor(sample)
        self._validate_messages(messages)

        full_text = self._tokenizer.apply_chat_template(messages)
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=True)

        if not self._logged_first_sample:
            logger.info(f"[SFT] First sample:\n{full_text}")
            self._logged_first_sample = True

        # Drop examples exceeding seq_len rather than truncating. Truncation
        # cuts the end of the response (often the final answer), which teaches
        # the model to produce incomplete outputs. Axolotl defaults to drop
        # for the same reason. If many examples are dropped, increase seq_len.
        if len(full_tokens) - 1 > self.seq_len:
            logger.debug(
                f"Dropping sample {self._sample_idx}: tokens exceeds seq_len {self.seq_len} \n {full_text}"
            )
            return None

        input_ids = full_tokens[:-1]
        label_ids = full_tokens[1:]

        # Find the prompt/response token boundary by tokenizing just the
        # user message with add_generation_prompt=True. This gives us the
        # prompt prefix including the assistant header. The delta between
        # this and the full tokenization is the assistant response.
        prompt_text = self._tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        prompt_len = len(prompt_tokens)

        # Mask prompt tokens in labels (accounting for the next-token shift:
        # label[i] predicts token i+1, so prompt labels span [0, prompt_len-1))
        mask_end = min(prompt_len - 1, len(label_ids))
        label_ids[:mask_end] = [IGNORE_INDEX] * mask_end

        return input_ids, label_ids

    def __iter__(self):
        yield from self._iter_greedy_packed()

    def _flush_buffers(self):
        """Pad buffers to seq_len, convert to tensors, clear, and return."""
        pad_len = self.seq_len - len(self._pack_buffer_input)
        if pad_len > 0:
            self._pack_buffer_input.extend([self._eos_id] * pad_len)
            self._pack_buffer_label.extend([IGNORE_INDEX] * pad_len)
            self._pack_buffer_positions.extend(range(pad_len))

        input_tensor = torch.tensor(self._pack_buffer_input, dtype=torch.long)
        label_tensor = torch.tensor(self._pack_buffer_label, dtype=torch.long)
        positions_tensor = torch.tensor(self._pack_buffer_positions, dtype=torch.long)
        self._pack_buffer_input = []
        self._pack_buffer_label = []
        self._pack_buffer_positions = []
        return {"input": input_tensor, "positions": positions_tensor}, label_tensor

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
                if result is None:
                    self._sample_idx += 1
                    continue

                input_ids, label_ids = result

                # If sample won't fit, pad and yield current buffer first
                if len(self._pack_buffer_input) + len(input_ids) > self.seq_len:
                    if self._pack_buffer_input:
                        yield self._flush_buffers()

                self._pack_buffer_input.extend(input_ids)
                self._pack_buffer_label.extend(label_ids)
                self._pack_buffer_positions.extend(range(len(input_ids)))
                self._sample_idx += 1

                # Yield if buffer is full, or immediately when not packing ("unpacked" mode)
                if not self.pack_sequences or len(self._pack_buffer_input) == self.seq_len:
                    yield self._flush_buffers()

            # Flush remaining buffer at end of data
            if self._pack_buffer_input:
                yield self._flush_buffers()

            if not self.infinite:
                logger.warning(f"SFT dataset '{self._dataset_id}' has run out of data")
                break
            else:
                self._sample_idx = 0
                self._epoch += 1
                self._data = self._data.shuffle(seed=42 + self._epoch)
                logger.warning(
                    f"SFT dataset '{self._dataset_id}' is being re-looped (epoch {self._epoch})"
                )

    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
            "epoch": self._epoch,
            "pack_buffer_input": list(self._pack_buffer_input),
            "pack_buffer_label": list(self._pack_buffer_label),
            "pack_buffer_positions": list(self._pack_buffer_positions),
        }

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._epoch = state_dict["epoch"]
        self._pack_buffer_input = list(state_dict["pack_buffer_input"])
        self._pack_buffer_label = list(state_dict["pack_buffer_label"])
        self._pack_buffer_positions = list(state_dict["pack_buffer_positions"])
        # Re-apply the epoch shuffle so resumed iteration sees the same
        # sample order as the original run.
        if self._epoch > 0:
            self._data = self._data.shuffle(seed=42 + self._epoch)


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
