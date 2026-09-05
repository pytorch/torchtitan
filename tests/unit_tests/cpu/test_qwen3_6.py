# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import pytest

pytest.importorskip("attn_gym")

from torchtitan.models.qwen3_5 import Qwen35Model, Qwen35StateDictAdapter
from torchtitan.models.qwen3_6 import model_registry, qwen3_6_configs
from torchtitan.models.qwen3_6.config_registry import qwen36_27b, qwen36_35b_a3b


def test_qwen36_registry_exposes_released_flavors() -> None:
    assert set(qwen3_6_configs) == {
        "debugmodel",
        "debugmodel_moe",
        "27B",
        "35B-A3B",
    }


@pytest.mark.parametrize("flavor", sorted(qwen3_6_configs))
def test_qwen36_registry_builds_every_flavor(flavor: str) -> None:
    model_spec = model_registry(
        flavor,
        moe_comm_backend=(
            "standard" if flavor == "debugmodel_moe" or "-A" in flavor else None
        ),
    )

    assert model_spec.name == "qwen3_6"
    assert model_spec.flavor == flavor
    assert model_spec.state_dict_adapter is Qwen35StateDictAdapter


def test_qwen36_27b_matches_hugging_face_config() -> None:
    config = cast(Qwen35Model.Config, model_registry("27B").model)

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
        model_registry("35B-A3B", moe_comm_backend="standard").model,
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


def test_qwen36_recipes_use_released_hugging_face_paths() -> None:
    dense_config = qwen36_27b()
    moe_config = qwen36_35b_a3b()

    assert dense_config.hf_assets_path.endswith("Qwen3.6-27B")
    assert dense_config.model_spec is not None
    assert dense_config.model_spec.name == "qwen3_6"
    assert moe_config.hf_assets_path.endswith("Qwen3.6-35B-A3B")
    assert moe_config.model_spec is not None
    assert moe_config.model_spec.name == "qwen3_6"
