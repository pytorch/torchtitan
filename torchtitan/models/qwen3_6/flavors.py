# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.6 model flavors."""

from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.qwen3_5.flavors import (
    _27b,
    _35b_a3b,
    _debugmodel,
    _debugmodel_moe,
    Qwen35Model,
    QWEN3_5_SPECIAL_TOKENS,
)

__all__ = [
    "build_model_config",
    "QWEN3_6_SPECIAL_TOKENS",
    "Qwen35Model",
    "MODEL_FLAVORS",
]

QWEN3_6_SPECIAL_TOKENS = dict(QWEN3_5_SPECIAL_TOKENS)

MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 4096),
    "debugmodel_moe": (_debugmodel_moe, 4096),
    "27B": (_27b, 262144),
    "35B-A3B": (_35b_a3b, 262144),
}


def build_model_config(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Qwen35Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter_config in converters:
            config = converter_config.build().convert(config)

    return config
