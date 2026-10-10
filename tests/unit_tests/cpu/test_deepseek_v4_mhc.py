# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for DeepSeek-V4's manifold-constrained hyper-connections."""

from unittest import mock

import pytest
import torch

from torchtitan.models.deepseek_v4.mhc import HcSplitSinkhorn


@pytest.mark.parametrize("sinkhorn_iters", [20, 1, 0])
def test_sinkhorn_per_entry_matches_tensor_form(sinkhorn_iters):
    """The compiled path's per-entry Sinkhorn equals eager's ``[T, hc, hc]`` loop, bitwise in fp64."""
    torch.manual_seed(0)
    sinkhorn = HcSplitSinkhorn.Config(hc_mult=4, sinkhorn_iters=sinkhorn_iters).build()
    mixes = torch.randn(64, (2 + 4) * 4, dtype=torch.float64)
    hc_scale = torch.rand(3, dtype=torch.float64)
    hc_base = torch.randn((2 + 4) * 4, dtype=torch.float64)

    expected = sinkhorn(mixes, hc_scale, hc_base)
    with mock.patch("torch.compiler.is_compiling", return_value=True):
        actual = sinkhorn(mixes, hc_scale, hc_base)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)
