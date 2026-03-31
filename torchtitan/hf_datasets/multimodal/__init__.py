# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import ClassVar

from torchtitan.components.tokenizer import HuggingFaceTokenizer


__all__ = ["MMSpecialTokens"]


@dataclass
class MMSpecialTokens:
    """Base class for multimodal special tokens.

    Subclasses define a ``SPECIAL_TOKENS_MAP`` mapping prefixes to token
    strings, e.g. ``{"img": "<|image_pad|>"}``.  ``from_tokenizer`` uses
    this map to:

    1. Look up each token string in the tokenizer to resolve its numeric ID.
    2. Set ``{prefix}_token`` (the string) and ``{prefix}_id`` (the int ID)
       on the instance.

    The ``_token`` fields are used in the data pipeline to build placeholder
    sequences in text (e.g. inserting ``<|image_pad|>`` tokens).  The ``_id``
    fields are used in the model to locate vision token positions in
    ``input_ids``.

    Subclasses must declare matching dataclass fields with defaults for each
    prefix (e.g. ``img_token: str = ""``, ``img_id: int = 0``) so that the
    dataclass constructor accepts them — defaults are required because the
    parent ``ignore_id`` field already has one.
    """

    # Subclasses must override with their own token map.
    SPECIAL_TOKENS_MAP: ClassVar[dict[str, str]] = {}

    ignore_id: int = -100  # PyTorch F.cross_entropy default

    @classmethod
    def from_tokenizer(cls, tokenizer: HuggingFaceTokenizer):
        token_map = cls.SPECIAL_TOKENS_MAP

        added_tokens = tokenizer.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}

        special_tokens_dict: dict[str, str | int] = {}
        for prefix, tok in token_map.items():
            if tok not in token_to_id:
                raise ValueError(
                    f"Special token '{tok}' not found in tokenizer's added tokens."
                )
            special_tokens_dict[f"{prefix}_token"] = tok
            special_tokens_dict[f"{prefix}_id"] = token_to_id[tok]

        return cls(**special_tokens_dict)
