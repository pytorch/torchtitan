# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torchtitan.tools.logging import logger
from transformers import AutoTokenizer


# HF AutoTokenizer will instantiate a root level logger, which will cause
# duplicate logs. We need to disable their root logger to avoid this.
def remove_notset_root_handlers():
    """
    Remove handlers with level NOTSET from root logger.
    Titan's logger is set, and thus we can differentiate between these.
    """
    for handler in logger.handlers[:]:
        if handler.level == logging.NOTSET:
            logger.removeHandler(handler)


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
