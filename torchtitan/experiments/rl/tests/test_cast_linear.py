# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``CastLinear`` and ``LMHeadCastConverter``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torchtitan.experiments.rl.models.cast_linear import CastLinear, LMHeadCastConverter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.qwen3 import qwen3_configs


def _qwen3_config():
    """A real decoder config tree to exercise the converter's traversal."""
    return qwen3_configs["0.6B"](attn_backend="flex")


def test_converter_swaps_only_lm_head():
    cfg = _qwen3_config()
    assert isinstance(cfg.lm_head, Linear.Config)
    assert not isinstance(cfg.lm_head, CastLinear.Config)
    # The model has many plain Linears besides the lm_head (attention, ffn).
    num_linears_before = sum(1 for _ in cfg.traverse(Linear.Config))
    assert num_linears_before > 1

    LMHeadCastConverter.Config().build().convert(cfg)

    # Only the lm_head was swapped to CastLinear.
    assert isinstance(cfg.lm_head, CastLinear.Config)
    casted = [fqn for fqn, _, _, _ in cfg.traverse(CastLinear.Config)]
    assert casted == ["lm_head"]
    # The swap doesn't add or drop Linear nodes (CastLinear.Config is a
    # Linear.Config subclass, so the total count is unchanged).
    assert sum(1 for _ in cfg.traverse(Linear.Config)) == num_linears_before


def test_converter_preserves_linear_fields():
    cfg = _qwen3_config()
    before = cfg.lm_head

    LMHeadCastConverter.Config().build().convert(cfg)

    after = cfg.lm_head
    assert after.in_features == before.in_features
    assert after.out_features == before.out_features
    assert after.bias == before.bias
    assert after.param_init is before.param_init
    assert after.sharding_config is before.sharding_config
    assert after.compute_dtype == "float32"


def test_forward_matmul_runs_in_fp32():
    lm_head = CastLinear.Config(in_features=8, out_features=16).build()
    lm_head = lm_head.to(torch.bfloat16)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    out = lm_head(x)

    # Output is fp32 while the stored weight stays bf16 -- the cast lives in
    # forward, so a tied embedding weight keeps its storage dtype.
    assert out.dtype == torch.float32
    assert lm_head.weight.dtype == torch.bfloat16

    # Bitwise-matches an explicit fp32 matmul of the same bf16 operands...
    ref_fp32 = F.linear(x.float(), lm_head.weight.float())
    assert torch.equal(out, ref_fp32)
    # ...and differs from a bf16-accumulated matmul, proving the cast matters.
    ref_bf16 = F.linear(x, lm_head.weight)
    assert not torch.equal(out, ref_bf16.float())


def test_compute_dtype_is_configurable():
    cfg = _qwen3_config()
    LMHeadCastConverter.Config(compute_dtype="bfloat16").build().convert(cfg)
    assert cfg.lm_head.compute_dtype == "bfloat16"

    lm_head = cfg.lm_head.build().to(torch.bfloat16)
    x = torch.randn(2, 4, cfg.lm_head.in_features, dtype=torch.bfloat16)
    out = lm_head(x)
    # compute_dtype=bfloat16 is a pure bf16 matmul (matches plain Linear).
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, F.linear(x, lm_head.weight))


def test_weight_tying_is_dtype_safe():
    # Emulate Decoder weight tying: the embedding shares the lm_head weight.
    lm_head = CastLinear.Config(in_features=8, out_features=16).build()
    lm_head = lm_head.to(torch.bfloat16)
    embedding = torch.nn.Embedding(16, 8).to(torch.bfloat16)
    embedding.weight = lm_head.weight

    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    out = lm_head(x)

    # The shared parameter is untouched (still bf16) even though the lm_head
    # matmul ran in fp32, so the embedding lookup is unaffected.
    assert out.dtype == torch.float32
    assert embedding.weight is lm_head.weight
    assert embedding.weight.dtype == torch.bfloat16


def test_converter_raises_when_lm_head_absent():
    # A config tree with no lm_head should be a hard error rather than a
    # silent no-op (e.g. if the output projection were ever renamed).
    cfg = _qwen3_config()
    cfg.lm_head = None
    with pytest.raises(ValueError, match="lm_head"):
        LMHeadCastConverter.Config().build().convert(cfg)
