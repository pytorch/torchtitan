# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# copied and adjusted from https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py

import os
from abc import ABC, abstractmethod
from typing import List
from logging import getLogger

from sentencepiece import SentencePieceProcessor


logger = getLogger()


class TokenizerIf(ABC):
    # tokenizer interface
    def __init__(self, tokenizer_path: str):
        assert os.path.exists(
            tokenizer_path
        ), f"The tokenizer path does not exist: {tokenizer_path}"
        assert os.path.isfile(tokenizer_path), tokenizer_path
        self._n_words = 8

    @abstractmethod
    def encode(self, *args, **kwargs) -> List[int]:
        ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str:
        ...

    @property
    def n_words(self) -> int:
        return self._n_words


def create_tokenizer(tokenizer_type: str, tokenizer_path: str) -> TokenizerIf:
    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {args.type}")


class SentencePieceTokenizer(TokenizerIf):
    """tokenizing and encoding/decoding text using SentencePiece."""
    def __init__(self, tokenizer_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            tokenizer_path (str): The path to the SentencePiece model file.
        """
        super().__init__(tokenizer_path)
        # reload tokenizer
        self.sp_model = SentencePieceProcessor(model_file=tokenizer_path)
        logger.info(f"Reloaded SentencePiece model from {tokenizer_path}")

        # BOS / EOS token IDs
        self._n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

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
        t = self.sp_model.encode(s)
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
        return self.sp_model.decode(t)
