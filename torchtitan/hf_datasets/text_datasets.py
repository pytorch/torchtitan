# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any

import numpy as np
import tyro
from datasets import load_dataset

from torchtitan.components.data.dataset import (
    DataRuntime,
    SampleProcessor,
    TokenSequence,
)
from torchtitan.components.loss import IGNORE_INDEX
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


class HuggingFaceTextProcessor(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        text_processor: Annotated[Callable, tyro.conf.Suppress]

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        self._tokenizer = runtime.tokenizer
        self._text_processor = config.text_processor

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TokenSequence:
        del rng
        # Use the dataset-specific text processor
        sample_text = self._text_processor(sample)
        sample_tokens = self._tokenizer.encode(sample_text, add_bos=True, add_eos=True)
        token_ids = np.asarray(sample_tokens, dtype=np.int64)
        return TokenSequence(
            token_ids=token_ids,
            loss_mask=np.ones(token_ids.shape, dtype=np.bool_),
        )


class ChatProcessor(SampleProcessor):
    """Tokenizes one single-turn chat sample and masks prompt labels."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        sample_processor: Annotated[Callable, tyro.conf.Suppress]

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        if runtime.tokenizer.eos_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_id set. "
                "ChatProcessor requires a tokenizer with a valid EOS token."
            )
        self._tokenizer = runtime.tokenizer
        self._eos_id = runtime.tokenizer.eos_id
        self.seq_len = runtime.seq_len
        self._sample_processor = config.sample_processor
        self._logged_first_sample = False

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

    def _tokenize_sample(self, sample: dict[str, Any]) -> TokenSequence | None:
        """Tokenize a single-turn sample and create input/label pairs.

        Returns token IDs and a loss mask with prompt tokens disabled.
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
            logger.info(f"[ChatProcessor] First sample full:\n{full_text}")
            self._logged_first_sample = True

        # Drop examples exceeding seq_len rather than truncating.
        if len(full_tokens) - 1 > self.seq_len:
            logger.debug(f"Dropping sample: tokens exceeds seq_len {self.seq_len}")
            return None

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

        loss_mask = np.concatenate(
            ([True], np.asarray(label_ids) != IGNORE_INDEX),
        )
        return TokenSequence(
            token_ids=np.asarray(full_tokens, dtype=np.int64),
            loss_mask=loss_mask,
        )

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TokenSequence | None:
        del rng
        return self._tokenize_sample(sample)
