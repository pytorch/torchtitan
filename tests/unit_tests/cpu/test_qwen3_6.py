# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import pytest

pytest.importorskip("attn_gym")

from torchtitan.models.qwen3_5 import Qwen35Model
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan.models.qwen3_6 import build_model_config, MODEL_FLAVORS


def test_qwen36_registry_exposes_released_flavors() -> None:
    assert set(MODEL_FLAVORS) == {
        "debugmodel",
        "debugmodel_moe",
        "27B",
        "35B-A3B",
    }


@pytest.mark.parametrize("flavor", sorted(MODEL_FLAVORS))
def test_qwen36_registry_builds_every_flavor(flavor: str) -> None:
    config = build_model_config(flavor)

    assert isinstance(config, Qwen35Model.Config)
    assert Qwen35Model.state_dict_adapter_cls is Qwen35StateDictAdapter


def test_qwen36_27b_matches_hugging_face_config() -> None:
    config = cast(Qwen35Model.Config, build_model_config("27B"))

    assert config.dim == 5120
    assert len(config.layers) == 64
    assert config.vision_encoder is not None
    assert config.vision_encoder.merger.fc2.out_features == 5120

    linear_layer = config.layers[0]
    assert linear_layer.delta_net is not None
    assert linear_layer.delta_net.in_proj_q.out_features == 16 * 128
    assert linear_layer.delta_net.in_proj_v.out_features == 48 * 128

    full_attention_layer = config.layers[3]
    assert full_attention_layer.attention is not None
    assert full_attention_layer.attention.n_heads == 24
    assert full_attention_layer.attention.n_kv_heads == 4


def test_qwen36_35b_a3b_matches_hugging_face_config() -> None:
    config = cast(
        Qwen35Model.Config,
        build_model_config("35B-A3B"),
    )

    assert config.dim == 2048
    assert len(config.layers) == 40
    assert config.vision_encoder is not None
    assert config.vision_encoder.merger.fc2.out_features == 2048

    linear_layer = config.layers[0]
    assert linear_layer.delta_net is not None
    assert linear_layer.delta_net.in_proj_q.out_features == 16 * 128
    assert linear_layer.delta_net.in_proj_v.out_features == 32 * 128
    assert linear_layer.moe is not None
    assert linear_layer.moe.router.num_experts == 256
    assert linear_layer.moe.router.top_k == 8

    full_attention_layer = config.layers[3]
    assert full_attention_layer.attention is not None
    assert full_attention_layer.attention.n_heads == 16
    assert full_attention_layer.attention.n_kv_heads == 2
