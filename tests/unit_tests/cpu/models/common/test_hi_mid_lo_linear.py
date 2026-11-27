# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.linear.hi_mid_lo_linear import (
    _split_into_bf16_pieces,
    HiMidLoLinear,
)


def test_hi_mid_lo_linear_raises_on_cpu():
    # The bf16-input, fp32-output GEMM is CUDA-only: FullPrecisionLinear is the CPU layer.
    layer = HiMidLoLinear.Config(in_features=8, out_features=4).build()
    layer = layer.to(dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="input is on cpu, not CUDA"):
        layer(torch.randn(6, 8, dtype=torch.bfloat16))


def test_hi_mid_lo_linear_preserves_linear_state_dict():
    layer = HiMidLoLinear.Config(
        in_features=8,
        out_features=4,
        bias=True,
    ).build()
    assert set(layer.state_dict()) == {"weight", "bias"}


def test_split_into_bf16_pieces_recovers_fp32():
    tensor = torch.randn(1000) * torch.logspace(-8, 8, 1000)

    hi, mid = _split_into_bf16_pieces(tensor, num_pieces=2, dim=0).chunk(2)

    assert hi.dtype == mid.dtype == torch.bfloat16
    relative_error = ((hi.float() + mid.float()) - tensor).abs() / tensor.abs()
    assert relative_error.max() <= 2**-15
    # One bf16 alone is ~100x worse.
    assert (tensor.bfloat16().float() - tensor).abs().div(tensor.abs()).max() > 2**-10
    # Three pieces are exact.
    pieces = _split_into_bf16_pieces(tensor, num_pieces=3, dim=0).chunk(3)
    assert sum(piece.double() for piece in pieces).equal(tensor.double())


def test_split_into_bf16_pieces_rounds_to_nearest():
    # 1 + 3 * 2^-9 is 0.75 of a bf16 step above 1, so hi rounds up and mid is negative.
    tensor = torch.tensor([1 + 3 * 2**-9])

    hi, mid, lo = _split_into_bf16_pieces(tensor, num_pieces=3, dim=0)

    assert hi.item() == 1 + 2**-7
    assert mid.item() == -(2**-9)
    assert lo.item() == 0.0


def test_lm_head_transform_swaps_only_lm_head():
    from torchtitan.config.transform import LMHeadFP32OutputTransform
    from torchtitan.models.muse_glimmer.model import SoftCappedLinear
    from torchtitan.models.qwen3 import MODEL_FLAVORS

    build_config, max_context_length = MODEL_FLAVORS["0.6B"]
    config = build_config(
        attn_backend="flex", seq_len=max_context_length, parallelism=ParallelismConfig()
    )
    lm_head_before = config.lm_head

    LMHeadFP32OutputTransform().transform(config)

    swapped = [
        fqn
        for fqn, linear_config, _, _ in config.traverse(Linear.Config)
        if isinstance(linear_config, HiMidLoLinear.Config)
    ]
    assert swapped == ["lm_head"]
    for field in fields(lm_head_before):
        assert getattr(config.lm_head, field.name) == getattr(
            lm_head_before, field.name
        )
    # The head keeps 2 pieces: summed over the vocab, a third doesn't help.
    assert config.lm_head.backward_mode == "hi_mid"
    # It turns on the split's local_compile region.
    assert "fp32_to_bf16_split" in config.local_compile.regions
    # Converting an already converted head is a no-op.
    converted = config.lm_head
    LMHeadFP32OutputTransform().transform(config)
    assert config.lm_head == converted
    # The flag is configurable.
    config.lm_head = lm_head_before
    LMHeadFP32OutputTransform(backward_mode="hi_mid_lo").transform(config)
    assert config.lm_head.backward_mode == "hi_mid_lo"
    # A later default pass keeps the converted head's backward_mode.
    LMHeadFP32OutputTransform().transform(config)
    assert config.lm_head.backward_mode == "hi_mid_lo"

    # A Linear subclass would lose its own fields and forward, e.g. a soft cap.
    config.lm_head = SoftCappedLinear.Config(
        **{
            field.name: getattr(lm_head_before, field.name)
            for field in fields(lm_head_before)
        },
        output_soft_cap_temp=30.0,
    )
    with pytest.raises(ValueError, match="is a SoftCappedLinear"):
        LMHeadFP32OutputTransform().transform(config)

    config.lm_head = None
    with pytest.raises(ValueError, match="lm_head"):
        LMHeadFP32OutputTransform().transform(config)
