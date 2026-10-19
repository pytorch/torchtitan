# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.models.kimi_k2_7 import build_model_config


# rms_norm_eps in the published config.json of moonshotai/Moonlight-16B-A3B,
# moonshotai/Kimi-VL-A3B-Instruct (text_config), and moonshotai/Kimi-K2.5
# (text_config). HF applies it to input_layernorm, post_attention_layernorm,
# and the final norm; the MLA q_a/kv_a layernorms use the 1e-6 module default.
@pytest.mark.parametrize("flavor", ["moonlight-16B-A3B", "Kimi-VL-A3B", "Kimi-K2.5"])
def test_norm_eps_matches_published_config(flavor: str) -> None:
    config = build_model_config(flavor, parallelism=ParallelismConfig())

    assert config.norm.eps == 1e-5
    for layer in config.layers:
        assert layer.attention_norm.eps == 1e-5
        assert layer.ffn_norm.eps == 1e-5
        assert layer.attention.kv_norm.eps == 1e-6


def test_deepseek_v3_keeps_its_norm_eps() -> None:
    config = build_deepseek_v3_model_config("671B", parallelism=ParallelismConfig())

    for layer in config.layers:
        assert layer.attention_norm.eps == 1e-6
        assert layer.ffn_norm.eps == 1e-6
