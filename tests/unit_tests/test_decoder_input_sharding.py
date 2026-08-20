# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
import torch
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import decoder_input_sharding


class _FakeParallelDims:
    cp_enabled = False


def test_decoder_input_sharding_matches_legacy_convention():
    sharding = decoder_input_sharding()
    assert set(sharding) == {"input", "labels", "positions"}
    for name in ("input", "positions"):
        t = sharding[name].per_axis_spmd_types()
        assert t[MeshAxisName.DP] == spmd.S(0)
        assert t[MeshAxisName.CP] == spmd.S(1)
        assert t[MeshAxisName.TP] == spmd.R
    labels = sharding["labels"].per_axis_spmd_types()
    assert labels[MeshAxisName.DP] == spmd.S(0)
    assert labels[MeshAxisName.CP] == spmd.S(1)
    assert labels[MeshAxisName.TP] == spmd.I


def test_decoder_build_forward_inputs_declares_sharding():
    dec = object.__new__(Decoder)
    dec.config = type("Cfg", (), {"first_attention": None})()
    input_dict = {"input": torch.zeros(2, 4), "positions": torch.arange(4).repeat(2, 1)}
    labels = torch.ones(2, 4)
    inputs, out_labels, extra, sharding = dec._build_forward_inputs(
        input_dict, labels, parallel_dims=_FakeParallelDims()
    )
    assert inputs is input_dict["input"]
    assert out_labels is labels
    assert sharding == decoder_input_sharding()
    assert "attention_masks" not in extra


def test_decoder_build_forward_inputs_builds_attention_masks():
    sentinel = object()
    dec = object.__new__(Decoder)
    first_attention = type("Attn", (), {"inner_attention": FlexAttention.Config()})()
    dec.config = type("Cfg", (), {"first_attention": first_attention})()
    dec.get_attention_masks = lambda positions: sentinel
    input_dict = {"input": torch.zeros(2, 4), "positions": torch.arange(4).repeat(2, 1)}
    labels = torch.ones(2, 4)
    _, _, extra, _ = dec._build_forward_inputs(
        input_dict, labels, parallel_dims=_FakeParallelDims()
    )
    assert extra["attention_masks"] is sentinel


def test_set_decoder_sharding_config_sets_root_sharding_and_no_input_sharding():
    from torchtitan.models.llama3 import llama3_configs
    from torchtitan.models.llama3.sharding import set_llama3_sharding_config

    config = llama3_configs["debugmodel"](attn_backend="flex")
    set_llama3_sharding_config(config, enable_sp=False)
    # It still does its real job: root-module sharding is populated.
    assert config.tok_embeddings.sharding_config is not None
    assert config.norm.sharding_config is not None
    assert config.lm_head.sharding_config is not None
    # The input_sharding config field is gone entirely.
    assert not hasattr(config, "input_sharding")
