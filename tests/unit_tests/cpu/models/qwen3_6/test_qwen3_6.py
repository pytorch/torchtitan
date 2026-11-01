# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import pytest
import torch

pytest.importorskip("attn_gym")

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention.gdn import GDN
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
    config = build_model_config(flavor, parallelism=ParallelismConfig())

    assert isinstance(config, Qwen35Model.Config)
    assert Qwen35Model.state_dict_adapter_cls is Qwen35StateDictAdapter


def test_qwen36_27b_matches_hugging_face_config() -> None:
    config = cast(
        Qwen35Model.Config, build_model_config("27B", parallelism=ParallelismConfig())
    )

    assert config.dim == 5120
    assert len(config.layers) == 64
    assert config.vision_encoder is not None
    assert config.vision_encoder.merger.fc2.out_features == 5120

    linear_layer = config.layers[0]
    assert isinstance(linear_layer.attention, GDN.Config)
    assert linear_layer.attention.n_key_heads == 16
    assert linear_layer.attention.n_heads == 48

    full_attention_layer = config.layers[3]
    assert full_attention_layer.attention is not None
    assert full_attention_layer.attention.n_heads == 24
    assert full_attention_layer.attention.n_kv_heads == 4


def test_qwen36_35b_a3b_matches_hugging_face_config() -> None:
    config = cast(
        Qwen35Model.Config,
        build_model_config("35B-A3B", parallelism=ParallelismConfig()),
    )

    assert config.dim == 2048
    assert len(config.layers) == 40
    assert config.vision_encoder is not None
    assert config.vision_encoder.merger.fc2.out_features == 2048

    linear_layer = config.layers[0]
    assert isinstance(linear_layer.attention, GDN.Config)
    assert linear_layer.attention.n_key_heads == 16
    assert linear_layer.attention.n_heads == 32
    assert linear_layer.feed_forward is not None
    assert linear_layer.feed_forward.router.num_experts == 256
    assert linear_layer.feed_forward.router.top_k == 8

    full_attention_layer = config.layers[3]
    assert full_attention_layer.attention is not None
    assert full_attention_layer.attention.n_heads == 16
    assert full_attention_layer.attention.n_kv_heads == 2


def test_adapter_packs_deltanet_projections_into_one_in_proj() -> None:
    config = cast(
        Qwen35Model.Config,
        build_model_config("debugmodel", parallelism=ParallelismConfig()),
    )
    adapter = Qwen35StateDictAdapter(config, hf_assets_path=None)
    delta_net = config.layers[0].attention
    assert isinstance(delta_net, GDN.Config)
    key_dim = delta_net.n_key_heads * delta_net.key_head_dim
    value_dim = delta_net.n_heads * delta_net.value_head_dim
    prefix = f"{adapter.hf_language_model_prefix}.layers.0.linear_attn"
    hf_state_dict = {
        f"{prefix}.in_proj_qkv.weight": torch.randn(
            2 * key_dim + value_dim, config.dim
        ),
        f"{prefix}.in_proj_z.weight": torch.randn(value_dim, config.dim),
        f"{prefix}.in_proj_a.weight": torch.randn(delta_net.n_heads, config.dim),
        f"{prefix}.in_proj_b.weight": torch.randn(delta_net.n_heads, config.dim),
        "lm_head.weight": torch.randn(2, 3),
    }

    native = adapter.from_hf(dict(hf_state_dict))
    in_proj = native.pop("layers.0.attention.in_proj.weight")
    assert native.keys() == {"lm_head.weight"}
    assert in_proj.shape == (delta_net.in_proj.out_features, config.dim)
    # Rows are grouped by key head: key head 1's q follows key head 0's whole group.
    group_rows = delta_net.in_proj.out_features // delta_net.n_key_heads
    head_dim = delta_net.key_head_dim
    assert torch.equal(
        in_proj[group_rows : group_rows + head_dim],
        hf_state_dict[f"{prefix}.in_proj_qkv.weight"][head_dim : 2 * head_dim],
    )

    restored = adapter.to_hf({**native, "layers.0.attention.in_proj.weight": in_proj})
    assert restored.keys() == hf_state_dict.keys()
    for key, value in hf_state_dict.items():
        assert torch.equal(restored[key], value), key
