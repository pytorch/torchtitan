# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.


import torch
from transformers import CLIPTokenizer, T5Tokenizer

from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer

from .configs import Encoder


class FluxTestTokenizer(BaseTokenizer):
    """
    Flux Tokenizer for test purpose. This is a simple wrapper around the TikTokenizer,
     to make it has same interface as the T5 and CLIP tokenizer used for Flux.
    """

    def __init__(self, model_path: str = "t5-small", max_length: int = 77, **hf_kwargs):
        self.tiktokenizer = HuggingFaceTokenizer(tokenizer_path=model_path, **hf_kwargs)
        self._max_length = max_length
        self.pad_id = 0

    def _pad_and_chunk_tokens(
        self, tokens: list[int], max_length: int, pad_token: int
    ) -> list[int]:
        # Pad the token sequence to max_length
        if len(tokens) < max_length:
            # If tokens are shorter than max_length, pad with pad_id or eos_id if pad_id is not defined
            padding = [pad_token] * (max_length - len(tokens))
            tokens = tokens + padding

        # Chunk the token sequence to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        return tokens

    def get_vocab_size(self) -> int:
        return self.tiktokenizer.vocab_size

    # pyrefly: ignore [bad-override]
    def encode(self, text: str | list[str]) -> torch.Tensor:
        """
        Use TikTokenizer to encode the text into tokens, and then pad and chunk the tokens to max_length.
        """
        if isinstance(text, list):
            if len(text) == 1:
                # for single item in list encode and add batch dimension
                tokens = self.tiktokenizer.encode(text[0], add_bos=True, add_eos=True)
                tokens = self._pad_and_chunk_tokens(
                    tokens, self._max_length, self.pad_id
                )
                return torch.tensor(tokens).unsqueeze(0)
            else:
                all_tokens = []
                for t in text:
                    tokens = self.tiktokenizer.encode(t, add_bos=True, add_eos=True)
                    tokens = self._pad_and_chunk_tokens(
                        tokens, self._max_length, self.pad_id
                    )
                    all_tokens.append(torch.tensor(tokens))
                return torch.stack(all_tokens)
        else:
            tokens = self.tiktokenizer.encode(text, add_bos=True, add_eos=True)
            tokens = self._pad_and_chunk_tokens(tokens, self._max_length, self.pad_id)
            return torch.tensor(tokens)

    # pyrefly: ignore [bad-override]
    def decode(self, t: list[int]) -> str:
        """
        Decode function. This function will not be called.
        """
        return self.tiktokenizer.decode(t)


class FluxTokenizer(BaseTokenizer):
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
            # pyrefly: ignore [bad-assignment]
            self._tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                model_path, max_length=max_length, **hf_kwargs
            )
        else:
            # pyrefly: ignore [bad-assignment]
            self._tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                model_path, max_length=max_length, **hf_kwargs
            )

    def get_vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    # pyrefly: ignore [bad-override]
    def encode(
        self,
        s: str | list[str],
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
            return_tensors="pt",  # return pytorch tensors, default return list[int]
        )["input_ids"]
        return tokens

    # pyrefly: ignore [bad-override]
    def decode(self, t: list[int]) -> list[str] | str:
        """
        Decode function. This function will not be called.
        """
        return self._tokenizer.decode(t)


def build_flux_tokenizer(
    encoder_config: Encoder,
    hf_assets_path: str,
) -> tuple[BaseTokenizer, BaseTokenizer]:
    """
    Build the tokenizer for Flux.
    """
    # pyrefly: ignore [missing-attribute]
    t5_tokenizer_path = encoder_config.t5_encoder
    # pyrefly: ignore [missing-attribute]
    clip_tokenzier_path = encoder_config.clip_encoder
    # pyrefly: ignore [missing-attribute]
    max_t5_encoding_len = encoder_config.max_t5_encoding_len

    # NOTE: This tokenizer is used for offline CI and testing only, borrowed from llama3 tokenizer
    # pyrefly: ignore [missing-attribute]
    if encoder_config.test_mode:
        tokenizer_class = FluxTestTokenizer
        t5_tokenizer_path = clip_tokenzier_path = hf_assets_path
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
