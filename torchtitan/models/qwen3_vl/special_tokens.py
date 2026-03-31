# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-VL special tokens and dataloader.

Qwen3VLSpecialTokens defines the mapping between token strings and IDs
for multimodal processing. It is used in two places:

- Data pipeline: the dataset and collator use _token fields (strings)
  to insert vision placeholder sequences into text, and pad_id / ignore_id
  for padding and label masking.

- Model forward: _id fields (ints) are used to locate vision token
  positions in input_ids for scattering vision embeddings and building
  MRoPE position IDs.

Qwen3VLDataLoader binds Qwen3VLSpecialTokens to the generic MMDataLoader.
Other VLMs define their own special tokens subclass and dataloader in the
same pattern.
"""

from dataclasses import dataclass
from typing import ClassVar

from torchtitan.hf_datasets.multimodal import MMSpecialTokens
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader


@dataclass
class Qwen3VLSpecialTokens(MMSpecialTokens):
    """Special tokens for Qwen3-VL multimodal processing."""

    SPECIAL_TOKENS_MAP: ClassVar[dict[str, str]] = {
        "img": "<|image_pad|>",
        "vid": "<|video_pad|>",
        "vision_start": "<|vision_start|>",
        "vision_end": "<|vision_end|>",
        "pad": "<|endoftext|>",
    }

    img_token: str = ""
    img_id: int = 0
    vid_token: str = ""
    vid_id: int = 0
    vision_start_token: str = ""
    vision_start_id: int = 0
    vision_end_token: str = ""
    vision_end_id: int = 0
    pad_token: str = ""
    pad_id: int = 0


class Qwen3VLDataLoader(MMDataLoader):
    """MMDataLoader configured with Qwen3-VL special tokens."""

    @dataclass(kw_only=True, slots=True)
    class Config(MMDataLoader.Config):
        pass

    special_tokens_cls = Qwen3VLSpecialTokens
