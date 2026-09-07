# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
import tyro

from torchtitan.components.data.dataset import (
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.components.data.sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
)
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.tools.logging import logger


def _read_text(sample: dict[str, Any]) -> str:
    return sample["text"]


class TextProcessor(SampleProcessor):
    """Tokenizes plain text into next-token input and label pairs."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        text_fn: Annotated[
            Callable[[dict[str, Any]], str], tyro.conf.Suppress
        ] = _read_text

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._tokenizer = context.tokenizer
        self._text_fn = config.text_fn

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        input_ids = np.asarray(
            self._tokenizer.encode(self._text_fn(sample), add_bos=True, add_eos=True),
            dtype=np.int64,
        )
        if len(input_ids) < 2:
            return None
        return TextSequence(
            input_ids=input_ids[:-1],
            labels=input_ids[1:],
        )


def _require_token_prefix(full_tokens: list[int], prompt_tokens: list[int]) -> None:
    """Raise if prompt_tokens is not an exact prefix of full_tokens."""
    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise ValueError(
            "Prompt tokens are not an exact prefix of the full conversation tokens"
        )


def _mask_prompt_labels(labels: np.ndarray, prompt_len: int, *, start: int = 0) -> None:
    """Ignore shifted labels for prompt tokens in [start, prompt_len).

    labels[i] is full_tokens[i + 1], so the single-turn formula
    labels[:max(prompt_len - 1, 0)] is the start=0 case.
    """
    labels[max(start - 1, 0) : max(prompt_len - 1, 0)] = IGNORE_INDEX


class ChatProcessor(SampleProcessor):
    """Tokenizes an alternating user/assistant conversation and masks prompt labels."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        messages_fn: Annotated[
            Callable[[dict[str, Any]], list[dict[str, str]]], tyro.conf.Suppress
        ]

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        if context.tokenizer.eos_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_id set. "
                "ChatProcessor requires a tokenizer with a valid EOS token."
            )
        self._tokenizer = context.tokenizer
        self._eos_id = context.tokenizer.eos_id
        self._max_context_length = context.max_context_length
        self._messages_fn = config.messages_fn
        self._logged_first_sample = False

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        """Validate even-length user/assistant alternation starting with user."""
        if len(messages) < 2 or len(messages) % 2 != 0:
            raise ValueError(
                "Expected an even-length alternating user/assistant "
                f"conversation, got {len(messages)} messages"
            )
        for index, message in enumerate(messages):
            expected_role = "user" if index % 2 == 0 else "assistant"
            role = message["role"]
            if role != expected_role:
                raise ValueError(
                    f"Expected messages[{index}] role '{expected_role}', got '{role}'"
                )

    def _encode_chat_messages(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        text = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt
        )
        return self._tokenizer.encode(text, add_bos=True, add_eos=False)

    def _tokenize_sample(self, sample: dict[str, Any]) -> TextSequence | None:
        """Tokenize a chat conversation and mask user-turn prompt labels.

        Returns None if the sample exceeds `seq_len`, avoiding
        training on truncated responses.

        Each user turn is re-tokenized with add_generation_prompt=True so the
        prompt/response boundary is taken from an exact token prefix.
        """
        messages = self._messages_fn(sample)
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

        # TODO(data-sft-overflow): Consider truncating oversized examples instead.
        # Causal loss remains valid for the retained response prefix.
        # Drop oversized examples rather than truncating.
        if len(full_tokens) - 1 > self._max_context_length:
            logger.debug(
                "Dropping sample: token count exceeds "
                f"max_context_length={self._max_context_length}"
            )
            return None

        tokens = np.asarray(full_tokens, dtype=np.int64)
        input_ids = tokens[:-1]
        labels = tokens[1:].copy()
        for turn in range(0, len(messages), 2):
            prompt_tokens = self._encode_chat_messages(
                messages[: turn + 1], add_generation_prompt=True
            )
            _require_token_prefix(full_tokens, prompt_tokens)
            start = 0
            if turn > 0:
                previous_tokens = self._encode_chat_messages(messages[:turn])
                _require_token_prefix(full_tokens, previous_tokens)
                start = len(previous_tokens)
            _mask_prompt_labels(labels, len(prompt_tokens), start=start)
        return TextSequence(
            input_ids=input_ids,
            labels=labels,
        )

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        return self._tokenize_sample(sample)


DATASETS: dict[str, SingleDatasetConfig] = {
    "c4": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="allenai/c4",
            name="en",
            split="train",
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    ),
    "c4_test": SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={
                "data_files": "tests/assets/c4_test/data.json",
            },
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    ),
    "c4_validation": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="allenai/c4",
            name="en",
            split="validation",
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    ),
}
