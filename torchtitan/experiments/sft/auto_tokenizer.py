# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from transformers import AutoTokenizer as HF_AutoTokenizer


class AutoTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_path: str):
        self.tokenizer = HF_AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.tokenizer_path = tokenizer_path
        self.chat_template = self.tokenizer.chat_template
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        if hasattr(self.tokenizer, "pad_token_id"):
            self.pad_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token
        else:
            self.pad_id = None
            self.pad_token = None

    def apply_chat_template(self, messages: list[dict], **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def encode(self, text: str, *args, **kwargs):
        return self.tokenizer.encode(text, *args, **kwargs)

    def decode(self, tokens: list[int], *args, **kwargs):
        return self.tokenizer.decode(tokens, *args, **kwargs)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def __call__(self, text: str, *args, **kwargs):
        return self.tokenizer(text, *args, **kwargs)


def build_auto_tokenizer(job_config: JobConfig) -> AutoTokenizer:
    return AutoTokenizer(job_config.model.hf_assets_path)
