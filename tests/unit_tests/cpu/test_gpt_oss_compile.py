# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.gpt_oss.moe import add_grouped_bias


@pytest.fixture(autouse=True)
def reset_local_compile():
    apply_local_compile([])
    torch._dynamo.reset()
    yield
    apply_local_compile([])


def _repeat_interleave_bias(output_RO, bias_EO, offsets_E):
    """Reference: expand each expert's bias over its rows; padding rows get zeros."""
    counts_E = torch.diff(torch.cat((offsets_E.new_zeros(1), offsets_E)))
    tail = output_RO.shape[0] - offsets_E[-1]
    bias_RO = torch.cat(
        (
            bias_EO.repeat_interleave(counts_E.long(), dim=0),
            bias_EO.new_zeros(tail, bias_EO.shape[-1]),
        )
    )
    return output_RO + bias_RO


@pytest.mark.parametrize("regions", [[], ["grouped_expert_bias"]])
def test_grouped_bias_matches_repeat_interleave(regions) -> None:
    torch.manual_seed(0)
    apply_local_compile(regions)
    # 3 experts own rows 0-1, (none), 2-4; rows 5-6 are padding.
    offsets_E = torch.tensor([2, 2, 5], dtype=torch.int32)
    grad_RO = torch.randn(7, 6, dtype=torch.float64)
    results = []
    for fn in (_repeat_interleave_bias, add_grouped_bias):
        output_RO = torch.randn(
            7, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
        )
        bias_EO = torch.randn(
            3, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
        )
        output_RO.requires_grad_(True)
        bias_EO.requires_grad_(True)
        out_RO = fn(output_RO, bias_EO, offsets_E)
        out_RO.backward(grad_RO)
        results.append((out_RO.detach(), output_RO.grad, bias_EO.grad))
    for reference, actual in zip(*results, strict=True):
        torch.testing.assert_close(actual, reference)
