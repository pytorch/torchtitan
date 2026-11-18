# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from unittest import mock

import pytest
import torch
from torch._inductor.graph import GraphLowering

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention.gdn import (
    _split_grouped_in_proj,
    _split_in_proj,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Per key head: [q | k | v | z | a | b]. 27B at TP4: 3 value heads per key head;
# 0.8B / 2B: 1 value head per key head. Both with 4 key heads.
SPLIT_SIZES = {
    "three_value_heads": [128, 128, 384, 384, 3, 3],
    "one_value_head": [128, 128, 128, 128, 1, 1],
}


def _triton_kernels(code: str) -> list[str]:
    return re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)


def _run_and_get_code_no_reset(fn):
    # run_and_get_code resets Dynamo first. Without the reset, the 37-token call sees
    # the 64-token compile and recompiles as the dynamic graph.
    codes: list[str] = []
    with mock.patch.object(GraphLowering, "save_output_code", codes.append):
        result = fn()
    return result, codes


@pytest.fixture
def split_region():
    torch._dynamo.reset()
    LocalCompileConfig(regions=["gdn_split_in_proj"]).apply()
    yield
    LocalCompileConfig().apply()
    torch._dynamo.reset()


@pytest.mark.parametrize("split_sizes", SPLIT_SIZES.values(), ids=SPLIT_SIZES.keys())
def test_split_in_proj_compiles_per_grad_mode(split_region, split_sizes) -> None:
    # 64 compiles a static graph, 37 the dynamic one, 1 a specialized one.
    for num_tokens in (64, 37, 1):
        projected_TC = torch.randn(
            num_tokens, 4 * sum(split_sizes), device="cuda", dtype=torch.bfloat16
        )
        expected = _split_grouped_in_proj(projected_TC, split_sizes, 128)

        # Inference: one combo kernel writes all four outputs.
        with torch.no_grad():
            outputs, codes = _run_and_get_code_no_reset(
                lambda: _split_in_proj(projected_TC, split_sizes, 128)
            )
        assert [len(_triton_kernels(code)) for code in codes] == [1], num_tokens
        assert all(torch.equal(o, e) for o, e in zip(outputs, expected))

        # Training: the backward copies each gradient into its slice (a foreach
        # kernel), not one kernel that picks a source per element.
        weights = [torch.randn(o.shape, device="cuda") for o in expected]
        outputs = []

        def loss_fn(split_fn, projected_TC):
            outputs[:] = split_fn(projected_TC, split_sizes, 128)
            return sum((o.float() * w).sum() for o, w in zip(outputs, weights))

        reference_TC = projected_TC.clone().requires_grad_()
        loss_fn(_split_grouped_in_proj, reference_TC).backward()
        projected_TC.requires_grad_()
        _, codes = _run_and_get_code_no_reset(
            lambda: loss_fn(_split_in_proj, projected_TC).backward()
        )
        assert len(codes) == 2, num_tokens
        for backward_code in codes[1:]:
            backward_kernels = _triton_kernels(backward_code)
            assert backward_kernels and all(
                name.startswith("triton_for_") for name in backward_kernels
            ), backward_kernels
        assert all(torch.equal(o, e) for o, e in zip(outputs, expected))
        assert torch.equal(projected_TC.grad, reference_TC.grad)
