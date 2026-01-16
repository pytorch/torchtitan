# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger
from transformers import AutoTokenizer as HF_AutoTokenizer


class HuggingFaceAutoTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer_path: str,
        eos_token: str,
        pad_token_id: int,
        pad_token: str | None = None,
    ):
        self.tokenizer = HF_AutoTokenizer.from_pretrained(
            tokenizer_path, eos_token=eos_token, use_fast=True
        )
        self.tokenizer_path = tokenizer_path
        self.chat_template = self.tokenizer.chat_template

        self.vocab_size = len(self.tokenizer)

        assert (
            pad_token_id < self.vocab_size
        ), f"PAD token ID is out of range: {pad_token_id} >= {self.vocab_size}"
        assert (
            pad_token_id != self.tokenizer.eos_token_id
        ), "PAD token ID is the same as EOS token ID, this can cause problems with varlen/flex attention in dynamic packing."

        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token

        self.pad_id = pad_token_id

        self.maybe_original_token_at_pad_id = self.tokenizer._convert_id_to_token(
            pad_token_id
        )
        if pad_token is not None:
            self.pad_token = pad_token
        else:
            self.pad_token = self.maybe_original_token_at_pad_id

        logger.info(
            f"[SFT AutoTokenizer] Using EOS token: {self.eos_token} - EOS ID: {self.eos_id} "
            f"[SFT AutoTokenizer] Using PAD token: {self.pad_token} - PAD ID: {self.pad_id}"
        )

    def apply_chat_template(self, messages: list[dict], **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def encode(self, text: str, *args, **kwargs):
        return self.tokenizer.encode(text, *args, **kwargs)

    def decode(self, tokens: list[int], *args, **kwargs):
        decoded = self.tokenizer.decode(tokens, *args, **kwargs)
        # this is an ad-hoc for debugging purpose
        if self.pad_id in tokens:
            decoded = decoded.replace(
                self.maybe_original_token_at_pad_id, self.pad_token
            )
        return decoded

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def __call__(self, text: str, *args, **kwargs):
        return self.tokenizer(text, *args, **kwargs)


def build_auto_tokenizer(job_config: JobConfig) -> HuggingFaceAutoTokenizer:
    eos_token = job_config.sft_config.eos_token
    pad_token_id = job_config.sft_config.pad_token_id
    pad_token = job_config.sft_config.pad_token
    assert (
        eos_token is not None and pad_token_id is not None
    ), "EOS and PAD token IDs must be provided"
    return HuggingFaceAutoTokenizer(
        job_config.model.hf_assets_path,
        eos_token=eos_token,
        pad_token_id=pad_token_id,
        pad_token=pad_token,
    )
