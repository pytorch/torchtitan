# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.models.llama3 import llama3_configs

from .model import FaultTolerantLlama3Model


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
) -> FaultTolerantLlama3Model.Config:
    get_config, max_context_len = llama3_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    base = get_config(attn_backend=attn_backend, seq_len=context_len)
    config = FaultTolerantLlama3Model.Config(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )
    return config
