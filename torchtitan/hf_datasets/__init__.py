# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass

from torchtitan.components.tokenizer import HuggingFaceTokenizer


__all__ = ["DatasetConfig", "SpecialTokens"]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable


@dataclass
class SpecialTokens:
    """Special tokens for multimodal processing."""

    img_token: str
    img_id: int
    vid_token: str
    vid_id: int
    vision_start_token: str
    vision_start_id: int
    vision_end_token: str
    vision_end_id: int
    pad_token: str
    pad_id: int
    ignore_id: int = -100  # PyTorch F.cross_entropy default

    @classmethod
    def from_tokenizer(cls, tokenizer: HuggingFaceTokenizer):
        SPECIAL_TOKENS_MAP = {
            "img": "<|image_pad|>",
            "vid": "<|video_pad|>",
            "vision_start": "<|vision_start|>",
            "vision_end": "<|vision_end|>",
            "pad": "<|endoftext|>",
        }
        added_tokens = tokenizer.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}

        # Try to get tokens from added tokens, fall back to encode if not found
        special_tokens_dict = {}
        for prefix, tok in SPECIAL_TOKENS_MAP.items():
            special_tokens_dict[f"{prefix}_token"] = tok
            if tok in token_to_id:
                special_tokens_dict[f"{prefix}_id"] = token_to_id[tok]
            else:
                # Fall back to encoding the token
                encoded = tokenizer.encode(tok)
                if len(encoded) != 1:
                    raise ValueError(
                        f"Special token '{tok}' encodes to {len(encoded)} tokens "
                        f"but must encode to exactly 1 token. "
                        f"Please use a tokenizer with the appropriate special tokens."
                    )
                special_tokens_dict[f"{prefix}_id"] = encoded[0]
        return cls(**special_tokens_dict)
