# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.


from typing import List

import torch
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer
from transformers import CLIPTokenizer, T5Tokenizer


class FluxTestTokenizer(Tokenizer):
    """
    Flux Tokenizer for test purpose. This is a simple wrapper around the TikTokenizer,
     to make it has same interface as the T5 and CLIP tokenizer used for Flux.
    """

    def __init__(self, model_path: str = "t5-small", max_length: int = 77, **hf_kwargs):
        self.tiktokenizer = TikTokenizer(model_path, **hf_kwargs)
        self._max_length = max_length
        self.pad_id = 0

    def _pad_and_chunk_tokens(
        self, tokens: List[int], max_length: int, pad_token: int
    ) -> List[int]:
        # Pad the token sequence to max_length
        if len(tokens) < max_length:
            # If tokens are shorter than max_length, pad with pad_id or eos_id if pad_id is not defined
            padding = [pad_token] * (max_length - len(tokens))
            tokens = tokens + padding

        # Chunk the token sequence to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        return tokens

    def encode(self, text: str) -> torch.Tensor:
        """
        Use TikTokenizer to encode the text into tokens, and then pad and chunk the tokens to max_length.
        """
        tokens = self.tiktokenizer.encode(text, bos=True, eos=True)
        tokens = self._pad_and_chunk_tokens(tokens, self._max_length, self.pad_id)
        return torch.tensor(tokens)

    def decode(self, t: List[int]) -> str:
        """
        Decode function. This function will not be called.
        """
        return self.tiktokenizer.decode(t)


class FluxTokenizer(Tokenizer):
    """
    Tokenizing and encoding/decoding text using the T5 or Clip tokenizer.

    Args:
        model_path (str): Path to the tokenzier from hugging face.

    """

    def __init__(self, model_path: str = "t5-small", max_length: int = 77, **hf_kwargs):
        super().__init__()
        self._n_words = 8  # TODO(jianiw): check
        self._max_length = max_length

        self.is_clip = "clip" in model_path.lower()

        if self.is_clip:
            self._tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                model_path, max_length=max_length, **hf_kwargs
            )
        else:
            self._tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                model_path, max_length=max_length, **hf_kwargs
            )

    def encode(
        self,
        s: str,
    ) -> torch.Tensor:
        """
        Encode the prompt text into tokens.
        """
        tokens = self._tokenizer(
            s,
            truncation=True,
            max_length=self._max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",  # return pytorch tensors, default return List[int]
        )["input_ids"]
        return tokens

    def decode(self, t: List[int]) -> str:
        """
        Decode function. This function will not be called.
        """
        return self._tokenizer.decode(t)


def build_flux_tokenizer(job_config: JobConfig) -> tuple[Tokenizer, Tokenizer]:
    """
    Build the tokenizer for Flux.
    """
    t5_tokenizer_path = job_config.encoder.t5_encoder
    clip_tokenzier_path = job_config.encoder.clip_encoder
    max_t5_encoding_len = job_config.encoder.max_t5_encoding_len

    # NOTE: This tokenizer is used for offline CI and testing only, borrowed from llama3 tokenizer
    if job_config.training.test_mode:
        tokenizer_class = FluxTestTokenizer
        t5_tokenizer_path = clip_tokenzier_path = job_config.model.tokenizer_path
    else:
        tokenizer_class = FluxTokenizer

    # T5 tokenzier will pad the token sequence to max_t5_encoding_len,
    # and CLIP tokenizer will pad the token sequence to 77 (fixed number).
    t5_tokenizer = tokenizer_class(
        t5_tokenizer_path,
        max_length=max_t5_encoding_len,
    )
    clip_tokenizer = tokenizer_class(
        clip_tokenzier_path,
        max_length=77,
    )

    return t5_tokenizer, clip_tokenizer
