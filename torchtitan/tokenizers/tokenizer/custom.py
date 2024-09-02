# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# copied and adjusted from https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py

from typing import List

from torchtitan.logging import logger
from transformers import AutoTokenizer


class CustomTokenizer:
    """
    Tokenizing and encoding/decoding text based on a Huggingface model.
    Args:
        tokenizer_path (str): The path to the Huggingface model file.
    """

    def __init__(self, tokenizer_path: str):

        # Load a tokenizer
        self.model = AutoTokenizer.from_pretrained(tokenizer_path)

        # Set config
        self.model.add_bos_token = False
        self.model.padding_side = "right"

        # BOS / EOS token IDs
        self._n_words: int = self.model.vocab_size
        self.bos_id: int = self.model.bos_token_id
        self.eos_id: int = self.model.eos_token_id
        self.pad_id: int = self.model.pad_token_id
        self.unk_id: int = self.model.unk_token_id
        logger.info(
            f"CustomTokenizer built: #words {self.n_words}, BOS ID {self.bos_id}, EOS ID {self.eos_id}"
        )

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str

        t = self.model.encode(s, add_special_tokens=False)

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.
        Args:
            t (List[int]): The list of token IDs to be decoded.
        Returns:
            str: The decoded string.
        """
        return self.model.decode(t)

    @property
    def n_words(self) -> int:
        return self._n_words
