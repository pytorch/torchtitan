# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from dataclasses import replace

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.feed_forward import DenseFeedForward, FeedForward
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.models.qwen3 import build_model_config


@pytest.mark.parametrize(
    "model_name,flavor",
    [
        ("llama3", "debugmodel"),
        ("deepseek_v3", "debugmodel"),
        ("gpt_oss", "debugmodel"),
        ("kimi_k3", "debugmodel"),
        ("qwen3", "debugmodel"),
        ("qwen3", "debugmodel_moe"),
        ("qwen3_5", "debugmodel"),
        ("qwen3_5", "debugmodel_moe"),
        ("qwen3_6", "debugmodel_moe"),
        ("qwen3_8", "debugmodel_moe"),
    ],
)
def test_feed_forward_registration_matches_config_paths(model_name, flavor):
    module = importlib.import_module(f"torchtitan.models.{model_name}")
    config = module.build_model_config(
        flavor,
        seq_len=128,
        attn_backend="flex",
        parallelism=ParallelismConfig(),
    )
    with torch.device("meta"):
        model = config.build()

    for layer_config, layer in zip(config.layers, model.layers.values(), strict=True):
        assert isinstance(layer_config.feed_forward, FeedForward.Config)
        assert isinstance(layer.feed_forward, FeedForward)
        expected_type = (
            MoE
            if isinstance(layer_config.feed_forward, MoE.Config)
            else DenseFeedForward
        )
        assert isinstance(layer.feed_forward, expected_type)
        assert not hasattr(layer, "moe")
        assert not hasattr(layer_config, "moe")

    # Checkpoint conversion and FQN-based transforms consume these paths.
    modules = dict(model.named_modules())
    for config_type in (
        FeedForward.Config,
        DenseFeedForward.Config,
        MoE.Config,
        GroupedLinear.Config,
    ):
        for fqn, child_config, parent, attr in config.traverse(config_type):
            assert fqn in modules
            assert getattr(parent, attr) is child_config


def test_first_feed_forward_skips_moe():
    config = build_deepseek_v3_model_config(
        "debugmodel", seq_len=128, parallelism=ParallelismConfig()
    )
    dense_layer = next(
        layer
        for layer in config.layers
        if isinstance(layer.feed_forward, DenseFeedForward.Config)
    )
    moe_layer = next(
        layer for layer in config.layers if isinstance(layer.feed_forward, MoE.Config)
    )

    config.layers = [moe_layer, dense_layer]
    assert config.first_feed_forward is dense_layer.feed_forward

    config.layers = [moe_layer]
    assert config.first_feed_forward is None


def test_moe_traversal_preserves_config_replacement_field():
    config = build_model_config(
        "debugmodel_moe", seq_len=128, parallelism=ParallelismConfig()
    )
    layer = config.layers[0]
    [(fqn, moe, parent, attr)] = list(layer.traverse(MoE.Config, _prefix="layers.0"))
    assert fqn == "layers.0.feed_forward"
    assert parent is layer
    assert attr == "feed_forward"
    replacement = replace(moe, load_balance_coeff=None)
    setattr(parent, attr, replacement)
    assert layer.feed_forward is replacement


def test_mtp_uses_feed_forward_for_moe():
    config = build_deepseek_v3_model_config(
        "debugmodel",
        seq_len=128,
        attn_backend="flex",
        parallelism=ParallelismConfig(),
        num_mtp_layers=1,
    )
    with torch.device("meta"):
        model = config.build()
    assert isinstance(model.mtp_layers[0].feed_forward, MoE)
    assert not hasattr(model.mtp_layers[0], "moe")
    assert "mtp_layers.0.feed_forward" in {
        fqn for fqn, *_ in config.traverse(MoE.Config)
    }
