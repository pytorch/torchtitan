# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_vlm
from torchtitan.models.qwen3_5 import (
    _27b,
    _35b_a3b,
    _debugmodel,
    _debugmodel_moe,
    parallelize_qwen3_5,
    Qwen35Model,
    QWEN3_5_SPECIAL_TOKENS,
)
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "model_registry",
    "QWEN3_6_SPECIAL_TOKENS",
    "Qwen35Model",
    "Qwen35StateDictAdapter",
    "qwen3_6_configs",
]

QWEN3_6_SPECIAL_TOKENS = dict(QWEN3_5_SPECIAL_TOKENS)

qwen3_6_configs = {
    "debugmodel": (_debugmodel, 4096),
    "debugmodel_moe": (_debugmodel_moe, 4096),
    "27B": (_27b, 262144),
    "35B-A3B": (_35b_a3b, 262144),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    get_config, max_context_len = qwen3_6_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
        **(
            {"moe_comm_backend": moe_comm_backend}
            if moe_comm_backend is not None
            else {}
        ),
    )
    if converters is not None:
        validate_converter_order(converters)
        for converter_config in converters:
            config = converter_config.build().convert(config)

    return ModelSpec(
        name="qwen3_6",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_vlm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen35StateDictAdapter,
    )
