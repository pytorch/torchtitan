# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class HFTransformers:
    model: str = ""
    """HuggingFace model ID (e.g., 'Qwen/Qwen3-4B-Instruct-2507')"""
    tie_word_embeddings: bool = False
    """Whether to tie input embeddings and output projection weights (default: True for HF models)"""


@dataclass
class JobConfig:
    hf_transformers: HFTransformers = field(default_factory=HFTransformers)
