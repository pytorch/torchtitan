# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import decoder_input_sharding


class _FakeParallelDims:
    cp_enabled = False


def test_decoder_build_forward_inputs_declares_sharding():
    dec = object.__new__(Decoder)
    dec.config = type("Cfg", (), {"first_attention": None})()
    input_dict = {
        "input": torch.zeros(2, 4),
        "labels": torch.ones(2, 4),
        "positions": torch.arange(4).repeat(2, 1),
    }
    out_dict, sharding = dec._build_forward_inputs(
        input_dict, parallel_dims=_FakeParallelDims()
    )
    assert out_dict["input"] is input_dict["input"]
    assert out_dict["labels"] is input_dict["labels"]
    assert sharding == decoder_input_sharding()
    assert "attention_masks" not in out_dict


def test_decoder_build_forward_inputs_builds_attention_masks():
    sentinel = object()
    dec = object.__new__(Decoder)
    first_attention = type("Attn", (), {"inner_attention": FlexAttention.Config()})()
    dec.config = type("Cfg", (), {"first_attention": first_attention})()
    dec.get_attention_masks = lambda positions: sentinel
    input_dict = {
        "input": torch.zeros(2, 4),
        "labels": torch.ones(2, 4),
        "positions": torch.arange(4).repeat(2, 1),
    }
    out_dict, _ = dec._build_forward_inputs(
        input_dict, parallel_dims=_FakeParallelDims()
    )
    assert out_dict["attention_masks"] is sentinel
