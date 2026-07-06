# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.experiments.transformers_modeling_backend.trainer import (
    HFTransformerTrainer,
)


@dataclass(kw_only=True, slots=True)
class TransformersBackendConfig(HFTransformerTrainer.Config):
    hf_model: str = ""
    """HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B')"""
