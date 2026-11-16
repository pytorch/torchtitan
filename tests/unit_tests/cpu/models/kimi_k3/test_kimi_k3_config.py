# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast

import torch

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan_recipes.overrides.kimi_attention_residual import TritonAttentionResidual
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


_TRITON_ATTENTION_RESIDUAL = (
    "torchtitan_recipes.overrides.kimi_attention_residual.triton_attention_residual"
)


def test_kimi_k3_enables_attention_residual_local_compile() -> None:
    config = kimi_k3_debugmodel()
    model_config = cast(KimiK3Model.Config, config.model)

    assert model_config.local_compile.regions == [
        "loss",
        "gated_rmsnorm",
        "fused_binary_activation",
        "fp32_to_bf16_split",
        "attention_residual",
        "mla_kv",
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


def test_preprocess_reuses_cpu_varlen_metadata(monkeypatch) -> None:
    positions = torch.tensor([0, 1, 0, 1, 2])
    varlen_metadata = create_varlen_metadata_for_document(positions)
    backend = type("Backend", (), {})
    model = object.__new__(KimiK3Model)
    model.config = SimpleNamespace(traverse=lambda *args, **kwargs: iter(()))
    model._get_attention_metadata = lambda **kwargs: {
        backend: LinearAttentionMetadata(
            varlen=varlen_metadata,
            num_conv_history_tokens=3,
        )
    }
    monkeypatch.setattr(
        "torchtitan.models.common.decoder.annotate_input_spmd_types",
        lambda context, inputs, shardings: inputs,
    )

    _, _, model_kwargs, _ = model.preprocess_inputs(
        {
            "input": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor([1, 2, 3, 4, 5]),
            "positions": positions,
            "varlen_metadata": varlen_metadata,
        },
        parallelism_context=SimpleNamespace(cp_enabled=False),
        parallelism=None,
    )

    metadata = model_kwargs["attention_metadata"][backend]
    assert metadata.varlen.cu_seq_q_cpu == [0, 2, 5]
    assert metadata.varlen.cu_seq_k_cpu == [0, 2, 5]
    assert metadata.num_conv_history_tokens == 3
