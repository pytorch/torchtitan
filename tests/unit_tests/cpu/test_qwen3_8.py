# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from typing import cast

import pytest
import torch

pytest.importorskip("attn_gym")

from torchtitan.models.qwen3_5 import Qwen35Model, Qwen35StateDictAdapter
from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config
from torchtitan.models.qwen3_8 import model_registry, qwen3_8_configs
from torchtitan.models.qwen3_8.config_registry import qwen38_27b, qwen38_2_4t_a95b


def test_qwen38_registry_exposes_only_qwen38_flavors() -> None:
    assert set(qwen3_8_configs) == {
        "debugmodel",
        "debugmodel_moe",
        "27B",
        "2.4T-A95B",
    }
    for legacy_flavor in (
        "0.8B",
        "2B",
        "4B",
        "9B",
        "35B-A3B",
        "122B-A10B",
        "397B-A17B",
    ):
        with pytest.raises(KeyError):
            model_registry(legacy_flavor)


def test_qwen38_27b_reuses_qwen35_multimodal_architecture() -> None:
    model_spec = model_registry("27B")
    config = cast(Qwen35Model.Config, model_spec.model)

    assert model_spec.name == "qwen3_8"
    assert config.dim == 5120
    assert len(config.layers) == 64
    assert config.vision_encoder is not None
    assert config.vision_encoder.merger.fc2.out_features == 5120


def test_qwen38_recipes_use_released_hugging_face_paths() -> None:
    dense_config = qwen38_27b()
    moe_config = qwen38_2_4t_a95b()

    assert dense_config.hf_assets_path.endswith("Qwen3.8-27B")
    assert dense_config.model_spec is not None
    assert dense_config.model_spec.name == "qwen3_8"
    assert moe_config.hf_assets_path.endswith("Qwen3.8-2.4T-A95B")
    assert moe_config.model_spec is not None
    assert moe_config.model_spec.name == "qwen3_8"


def test_qwen38_2_4t_a95b_matches_hugging_face_config() -> None:
    build_config, max_context_length = qwen3_8_configs["2.4T-A95B"]
    config = build_config(
        attn_backend="flex",
        moe_comm_backend="standard",
        seq_len=max_context_length,
    )

    assert config.dim == 8192
    assert len(config.layers) == 92
    assert config.vision_encoder is None

    linear_layer = config.layers[0]
    assert linear_layer.delta_net is not None
    assert linear_layer.delta_net.in_proj_q.out_features == 16 * 128
    assert linear_layer.delta_net.in_proj_v.out_features == 128 * 128

    full_attention_layer = config.layers[3]
    assert full_attention_layer.attention is not None
    assert full_attention_layer.attention.n_heads == 64
    assert full_attention_layer.attention.n_kv_heads == 4

    assert linear_layer.moe is not None
    assert linear_layer.moe.router.num_experts == 512
    assert linear_layer.moe.router.top_k == 10


def test_text_only_qwen38_sharding_does_not_require_vision() -> None:
    build_config, max_context_length = qwen3_8_configs["2.4T-A95B"]
    config = build_config(
        attn_backend="flex",
        moe_comm_backend="standard",
        seq_len=max_context_length,
    )

    set_qwen35_sharding_config(config, enable_sp=True, enable_ep=True)

    assert config.tok_embeddings.sharding_config is not None
    assert config.layers[0].sharding_config is not None


def test_shared_model_builds_without_vision_encoder() -> None:
    build_config, max_context_length = qwen3_8_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    config = replace(
        config,
        vocab_size=128,
        tok_embeddings=replace(config.tok_embeddings, num_embeddings=128),
        lm_head=replace(config.lm_head, out_features=128),
        vision_encoder=None,
    )

    model = config.build()

    assert model.vision_encoder is None


def test_text_only_checkpoint_adapter_uses_model_prefix() -> None:
    build_config, max_context_length = qwen3_8_configs["2.4T-A95B"]
    config = build_config(
        attn_backend="flex",
        moe_comm_backend="standard",
        seq_len=max_context_length,
    )
    adapter = Qwen35StateDictAdapter(config, hf_assets_path=None)
    embedding = torch.randn(2, 3)
    lm_head = torch.randn(2, 3)

    converted = adapter.from_hf(
        {
            "model.embed_tokens.weight": embedding,
            "lm_head.weight": lm_head,
        }
    )
    assert set(converted) == {"tok_embeddings.weight", "lm_head.weight"}
    torch.testing.assert_close(converted["tok_embeddings.weight"], embedding)
    torch.testing.assert_close(converted["lm_head.weight"], lm_head)

    restored = adapter.to_hf(converted)
    assert set(restored) == {"model.embed_tokens.weight", "lm_head.weight"}
    torch.testing.assert_close(restored["model.embed_tokens.weight"], embedding)
    torch.testing.assert_close(restored["lm_head.weight"], lm_head)


def test_multimodal_checkpoint_adapter_keeps_language_model_prefix() -> None:
    build_config, max_context_length = qwen3_8_configs["27B"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    adapter = Qwen35StateDictAdapter(config, hf_assets_path=None)
    embedding = torch.randn(2, 3)
    lm_head = torch.randn(2, 3)

    converted = adapter.from_hf(
        {
            "model.language_model.embed_tokens.weight": embedding,
            "lm_head.weight": lm_head,
        }
    )
    restored = adapter.to_hf(converted)

    assert "model.language_model.embed_tokens.weight" in restored
    assert "model.embed_tokens.weight" not in restored


def test_text_only_checkpoint_adapter_converts_fused_deltanet_qkv() -> None:
    build_config, max_context_length = qwen3_8_configs["2.4T-A95B"]
    config = build_config(
        attn_backend="flex",
        moe_comm_backend="standard",
        seq_len=max_context_length,
    )
    adapter = Qwen35StateDictAdapter(config, hf_assets_path=None)
    delta_net = config.layers[0].delta_net
    assert delta_net is not None
    key_dim = delta_net.in_proj_q.out_features
    value_dim = delta_net.in_proj_v.out_features
    fused_qkv = torch.randn(key_dim * 2 + value_dim, 1)

    converted = adapter.from_hf(
        {
            "model.layers.0.linear_attn.in_proj_qkv.weight": fused_qkv,
            "lm_head.weight": torch.randn(2, 3),
        }
    )
    assert set(converted) == {
        "layers.0.attn.in_proj_q.weight",
        "layers.0.attn.in_proj_k.weight",
        "layers.0.attn.in_proj_v.weight",
        "lm_head.weight",
    }

    restored = adapter.to_hf(converted)
    torch.testing.assert_close(
        restored["model.layers.0.linear_attn.in_proj_qkv.weight"],
        fused_qkv,
    )
