# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tokenizers import AddedToken

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config.job_config import JobConfig
from torchtitan.tools.logging import logger


class VLMTokenizer(HuggingFaceTokenizer):
    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)
        self.pad_token = "<|pad|>"
        self.img_token = "<|image|>"
        self.boi_token = "<|begin_of_image|>"
        self.eoi_token = "<|end_of_image|>"
        _special_tokens = [
            AddedToken(token_str, special=True)
            for token_str in [
                self.pad_token,
                self.img_token,
                self.boi_token,
                self.eoi_token,
            ]
        ]
        num_new_tokens = self.tokenizer.add_special_tokens(_special_tokens)
        logger.info(f"{num_new_tokens} new tokens were added to the tokenizer.")

        self.pad_id = self.tokenizer.token_to_id(self.pad_token)
        self.img_id = self.tokenizer.token_to_id(self.img_token)
        self.boi_id = self.tokenizer.token_to_id(self.boi_token)
        self.eoi_id = self.tokenizer.token_to_id(self.eoi_token)


def build_vlm_tokenizer(job_config: JobConfig) -> VLMTokenizer:
    tokenizer = VLMTokenizer(job_config.model.hf_assets_path)
    return tokenizer
