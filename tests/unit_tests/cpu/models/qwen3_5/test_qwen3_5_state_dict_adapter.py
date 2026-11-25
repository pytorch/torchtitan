# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import cast

import pytest
import torch

pytest.importorskip("attn_gym")

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.qwen3_5 import build_model_config, Qwen35Model
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter


def test_adapter_packs_deltanet_projections_into_one_in_proj() -> None:
    config = cast(
        Qwen35Model.Config,
        build_model_config("debugmodel_mm", parallelism=ParallelismConfig()),
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
    q, k, v = hf_state_dict[f"{prefix}.in_proj_qkv.weight"].split(
        [key_dim, key_dim, value_dim]
    )
    z, a, b = (hf_state_dict[f"{prefix}.in_proj_{name}.weight"] for name in "zab")
    # Key head g's rows: [q_g | k_g | v_g | z_g | a_g | b_g] (GDN.Config).
    expected = torch.cat(
        [
            torch.cat([t.chunk(delta_net.n_key_heads)[g] for t in (q, k, v, z, a, b)])
            for g in range(delta_net.n_key_heads)
        ]
    )
    assert torch.equal(in_proj, expected)

    restored = adapter.to_hf({**native, "layers.0.attention.in_proj.weight": in_proj})
    assert restored.keys() == hf_state_dict.keys()
    for key, value in hf_state_dict.items():
        assert torch.equal(restored[key], value), key


def test_text_only_flavor_loads_multimodal_checkpoint(tmp_path) -> None:
    """A text-only flavor reads the decoder from a multimodal HF checkpoint
    (``model.language_model.*``) and ignores its vision weights."""
    weight_map = {
        "model.language_model.embed_tokens.weight": "model-00001.safetensors",
        "model.language_model.norm.weight": "model-00001.safetensors",
        "model.visual.merger.norm.weight": "model-00001.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    config = cast(
        Qwen35Model.Config,
        build_model_config("debugmodel", parallelism=ParallelismConfig()),
    )
    adapter = Qwen35StateDictAdapter(config, hf_assets_path=str(tmp_path))
    assert adapter.hf_language_model_prefix == "model.language_model"

    native = {"tok_embeddings.weight": torch.randn(2, 3), "norm.weight": torch.randn(3)}
    hf_state_dict = adapter.to_hf(native)
    assert hf_state_dict.keys() == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
    hf_state_dict["model.visual.merger.norm.weight"] = torch.randn(3)
    restored = adapter.from_hf(hf_state_dict)
    assert restored.keys() == native.keys() | {"lm_head.weight"}

    # Without an index, a text-only flavor assumes a text-only checkpoint.
    adapter = Qwen35StateDictAdapter(config, hf_assets_path=None)
    assert adapter.hf_language_model_prefix == "model"
