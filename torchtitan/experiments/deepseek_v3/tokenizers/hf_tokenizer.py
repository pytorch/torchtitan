# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.tools.logging import logger
from transformers import AutoTokenizer


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, bos=False, eos=False, **kwargs):
        # Handle bos and eos parameters
        if bos:
            kwargs["add_special_tokens"] = True
        if eos:
            kwargs["add_special_tokens"] = True

        return self.tokenizer.encode(text, **kwargs)

    def __getattr__(self, name):
        # Delegate all other attributes/methods to the underlying tokenizer
        return getattr(self.tokenizer, name)


def get_hf_tokenizer(model_id: str):
    logger.info(f"Instantiating tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return TokenizerWrapper(tokenizer)
