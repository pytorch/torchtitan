# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline, stateful packed token fixture for training-chain verification."""

from dataclasses import dataclass

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer


class FixedTokenTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_size: int = 64

    def __init__(self, config, *, tokenizer_path=None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.eos_id = 2

    def encode(self, text, **kwargs):
        return [ord(char) % self.vocab_size for char in text]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))

    def get_vocab_size(self):
        return self.vocab_size


class PackedTokenDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        document_length: int = 32

    def __init__(
        self,
        config,
        *,
        tokenizer,
        seq_len,
        local_batch_size,
        dp_rank,
        dp_world_size,
        **kwargs,
    ):
        if local_batch_size != 1 or dp_world_size != 1:
            raise ValueError("The offline draft loader supports one single-rank row")
        if config.document_length % 2 or seq_len % config.document_length:
            raise ValueError("Documents must be even and tile seq_len")
        self.seq_len = seq_len
        self.document_length = config.document_length
        self.vocab_size = tokenizer.get_vocab_size()
        self.cursor = 0

    def __iter__(self):
        while True:
            offsets = torch.arange(self.seq_len)
            tokens = (offsets + self.cursor * 3).remainder(self.vocab_size).unsqueeze(0)
            labels = (tokens + 1).remainder(self.vocab_size)
            labels[:, self.document_length - 1 :: self.document_length] = -100
            positions = offsets.remainder(self.document_length).unsqueeze(0)
            self.cursor += 1
            yield {"input": tokens, "positions": positions}, labels

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
