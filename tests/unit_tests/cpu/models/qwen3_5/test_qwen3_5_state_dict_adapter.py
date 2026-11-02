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
from torchtitan.models.qwen3_5 import build_model_config, Qwen35Model
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter


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
