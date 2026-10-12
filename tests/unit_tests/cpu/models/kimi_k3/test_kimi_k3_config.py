# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan_recipes.overrides.kimi_attention_residual import TritonAttentionResidual
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


_TRITON_ATTENTION_RESIDUAL = (
    "torchtitan_recipes.overrides.kimi_attention_residual." "triton_attention_residual"
)


def test_kimi_k3_enables_attention_residual_local_compile() -> None:
    config = kimi_k3_debugmodel()
    model_config = cast(KimiK3Model.Config, config.model)

    assert model_config.local_compile_regions == [
        "loss",
        "gated_rmsnorm",
        "fused_binary_activation",
        "fp32_to_bf16_split",
        "attention_residual",
    ]


def test_triton_attention_residual_override_replaces_all_configs() -> None:
    config = kimi_k3_debugmodel()
    model_config = cast(KimiK3Model.Config, config.model)

    replacements = apply_overrides(
        OverrideConfig(imports=[_TRITON_ATTENTION_RESIDUAL]),
        config,
    )

    assert len(replacements) == len(model_config.layers) + 1
    assert isinstance(
        model_config.attention_residual,
        TritonAttentionResidual.Config,
    )
    assert all(
        isinstance(
            layer_config.attention_residual,
            TritonAttentionResidual.Config,
        )
        for layer_config in model_config.layers
    )
