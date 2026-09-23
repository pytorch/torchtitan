# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.models.qwen3 import qwen3_configs

from ..common_utils import build_decoder_config_for_backend
from .model import GraphTrainerQwen3Model


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
) -> GraphTrainerQwen3Model.Config:
    kwargs = {}
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    get_config, max_context_len = qwen3_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    base = build_decoder_config_for_backend(
        get_config, attn_backend, seq_len=context_len, **kwargs
    )
    config = GraphTrainerQwen3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return config
