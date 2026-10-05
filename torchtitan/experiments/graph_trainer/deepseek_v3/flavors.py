# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer DeepSeek V3 model flavors."""

from dataclasses import fields

from torchtitan.models.deepseek_v3 import MODEL_FLAVORS

from ..common_utils import build_decoder_config_for_backend
from .model import GraphTrainerDeepSeekV3Model


def build_model_config(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
) -> GraphTrainerDeepSeekV3Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    base = build_decoder_config_for_backend(
        get_config,
        attn_backend,
        seq_len=context_len,
    )
    config = GraphTrainerDeepSeekV3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return config
