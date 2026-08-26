# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torch import nn

from torchtitan.experiments.rl.models.native_vllm_qwen3_5 import (
    build_qwen35_native_weight_views,
    qwen35_text_state_dict,
)


def _weight(rows: int, columns: int = 4) -> nn.Module:
    module = nn.Module()
    module.weight = nn.Parameter(torch.zeros(rows, columns))
    return module


def _norm(size: int = 4) -> nn.Module:
    module = nn.Module()
    module.weight = nn.Parameter(torch.zeros(size))
    return module


def _feed_forward_config() -> SimpleNamespace:
    return SimpleNamespace(
        w1=SimpleNamespace(out_features=16),
        w3=SimpleNamespace(out_features=16),
    )


def _native_layer() -> nn.Module:
    layer = nn.Module()
    layer.input_layernorm = _norm()
    layer.post_attention_layernorm = _norm()
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = _weight(8)
    layer.mlp.down_proj = _weight(4, 4)
    return layer


def test_native_qwen35_weight_views_cover_packed_full_and_gdn_layers():
    full_native = _native_layer()
    full_native.self_attn = nn.Module()
    full_native.self_attn.qkv_proj = _weight(8)
    full_native.self_attn.o_proj = _weight(4, 4)
    full_native.self_attn.q_norm = _norm(2)
    full_native.self_attn.k_norm = _norm(2)
    full_config = SimpleNamespace(
        feed_forward=_feed_forward_config(),
        moe=None,
        attention=SimpleNamespace(
            wq=SimpleNamespace(out_features=16),
            wk=SimpleNamespace(out_features=8),
            wv=SimpleNamespace(out_features=8),
        ),
        delta_net=None,
    )

    gdn_native = _native_layer()
    gdn_native.linear_attn = nn.Module()
    gdn_native.linear_attn.in_proj_qkvz = _weight(20)
    gdn_native.linear_attn.in_proj_ba = _weight(6)
    gdn_native.linear_attn.conv1d = _weight(14, 2)
    gdn_native.linear_attn.A_log = nn.Parameter(torch.zeros(3))
    gdn_native.linear_attn.dt_bias = nn.Parameter(torch.zeros(3))
    gdn_native.linear_attn.norm = _norm(2)
    gdn_native.linear_attn.out_proj = _weight(4, 6)
    gdn_config = SimpleNamespace(
        feed_forward=_feed_forward_config(),
        moe=None,
        attention=None,
        delta_net=SimpleNamespace(
            in_proj_q=SimpleNamespace(out_features=16),
            in_proj_k=SimpleNamespace(out_features=16),
            in_proj_v=SimpleNamespace(out_features=24),
            in_proj_z=SimpleNamespace(out_features=24),
            in_proj_b=SimpleNamespace(out_features=12),
            in_proj_a=SimpleNamespace(out_features=12),
        ),
    )

    core = nn.Module()
    core.embed_tokens = _weight(8)
    core.norm = _norm()
    core.layers = nn.ModuleList([full_native, gdn_native])
    language_model = nn.Module()
    language_model.model = core
    language_model.lm_head = _weight(8)
    native_model = nn.Module()
    native_model.language_model = language_model
    model_spec = SimpleNamespace(
        name="qwen3_5",
        model=SimpleNamespace(layers=[full_config, gdn_config]),
    )

    views, layouts = build_qwen35_native_weight_views(
        native_model,
        model_spec,
        tensor_parallel_size=4,
    )

    assert set(views) == set(layouts)
    assert views["layers.0.attn.wq.weight"].shape == (4, 4)
    assert views["layers.0.attn.wk.weight"].shape == (2, 4)
    assert views["layers.0.attn.wv.weight"].shape == (2, 4)
    assert views["layers.1.attn.in_proj_q.weight"].shape == (4, 4)
    assert views["layers.1.attn.in_proj_v.weight"].shape == (6, 4)
    assert views["layers.1.attn.in_proj_z.weight"].shape == (6, 4)
    assert views["layers.1.attn.in_proj_b.weight"].shape == (3, 4)
    assert views["layers.1.attn.in_proj_a.weight"].shape == (3, 4)
    assert views["layers.1.attn.conv_v.weight"].shape == (6, 2)

    views["layers.0.attn.wk.weight"].fill_(7)
    assert torch.equal(
        full_native.self_attn.qkv_proj.weight[4:6],
        torch.full((2, 4), 7.0),
    )
    views["layers.1.attn.in_proj_a.weight"].fill_(5)
    assert torch.equal(
        gdn_native.linear_attn.in_proj_ba.weight[3:6],
        torch.full((3, 4), 5.0),
    )


def test_qwen35_text_state_dict_drops_vision_weights():
    state_dict = {
        "tok_embeddings.weight": 1,
        "layers.0.attn.wq.weight": 2,
        "norm.weight": 3,
        "lm_head.weight": 4,
        "vision.patch_embed.weight": 5,
    }

    assert qwen35_text_state_dict(state_dict) == {
        "tok_embeddings.weight": 1,
        "layers.0.attn.wq.weight": 2,
        "norm.weight": 3,
        "lm_head.weight": 4,
    }
