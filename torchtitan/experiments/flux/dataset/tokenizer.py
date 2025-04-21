# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.


from typing import List

from torchtitan.components.tokenizer import Tokenizer
from transformers import CLIPTokenizer, T5Tokenizer


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
    ) -> List[int]:
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
