# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SFTConfig:
    split: str = "train"
    """Split to use"""
    dataset_subset: str | None = None
    """Subset to use"""
    stream_dataset: bool = True
    """Whether to stream the dataset"""
    """
    At moment we support two types of datasets:
    # about dataset
    1. **Multi-turn (`True`):** Loads data from the column specified by `messages_key`.
        The content must be a list of dictionaries containing 'role' and 'content'.
        Example:
        ```
        [
            {
                "role": "user",
                "content": "Hello, how are you?"
            },
            {
                "role": "assistant",
                "content": "I am good, thank you!"
            }
        ]
        ```
    2. **Single-turn (`False`):** Loads data from two separate columns
        specified by `prompt_key` and `response_key`.

    """

    apply_chat_template: bool = False
    """
    If True, we will apply chat template to the messages, otherwise we will use the raw messages.
    It is important that chat_templated should be properly configured for the tokenizer and
    it can render the messages for multi-turn mode.
    """
    pad_mode: Literal["right_padding", "greedy_packing"] = "greedy_packing"
    """
    Padding mode to use.
    - "right": One batch contains only one unique sequence, the sequence will be padded to the right to reach the max length.
    - "greedy_packing": One batch contains multiple short sequences and then be padded to the right to reach the max length.
    """
    chat_template_kwargs: dict = field(default_factory=dict)
    """
    When apply chat template, we will pass these kwargs to the tokenizer.apply_chat_template function.
    It has to be HF's apply_chat_template kwargs.
    """
    ignore_input_ids_mismatch: bool = False
    """
    Whether to ignore the input_ids mismatch when apply chat template.
    """

    eos_token: str | None = None
    """
    EOS token to use.
    """
    pad_token_id: int | None = None
    """
    PAD token ID to use.
    """
    pad_token: str | None = None
    """
    PAD token to override the tokenizer's pad token.
    """


@dataclass
class JobConfig:
    sft_config: SFTConfig = field(default_factory=SFTConfig)
