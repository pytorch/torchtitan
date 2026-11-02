# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

import pytest
import torch

pytest.importorskip("attn_gym")

from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention.gdn import (
    _split_grouped_in_proj,
    _split_in_proj,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# 27B at TP4: 4 key heads x [q 128 | k 128 | v 384 | z 384 | a 3 | b 3].
SPLIT_SIZES = [128, 128, 384, 384, 3, 3]


def _triton_kernels(code: str) -> list[str]:
    return re.findall(r"^(triton_\w+) = async_compile\.triton", code, re.MULTILINE)


@pytest.fixture
def split_region():
    LocalCompileConfig(regions=["gdn_split_in_proj"]).apply()
    yield
    LocalCompileConfig().apply()
    torch._dynamo.reset()


def test_split_in_proj_compiles_per_grad_mode(split_region) -> None:
    projected_TC = torch.randn(
        64, 4 * sum(SPLIT_SIZES), device="cuda", dtype=torch.bfloat16
    )
    expected = _split_grouped_in_proj(projected_TC, SPLIT_SIZES, 128)

    # Inference: one combo kernel writes all four outputs.
    with torch.no_grad():
        outputs, codes = run_and_get_code(
            _split_in_proj, projected_TC, SPLIT_SIZES, 128
        )
    assert [len(_triton_kernels(code)) for code in codes] == [1]
    assert all(torch.equal(o, e) for o, e in zip(outputs, expected))

    # Training: the backward copies each gradient into its slice (a foreach kernel),
    # not one kernel that picks a source per element.
    weights = [torch.randn(o.shape, device="cuda") for o in expected]

    def loss_fn(split_fn, projected_TC):
        outputs = split_fn(projected_TC, SPLIT_SIZES, 128)
        return sum((o.float() * w).sum() for o, w in zip(outputs, weights))

    reference_TC = projected_TC.clone().requires_grad_()
    loss_fn(_split_grouped_in_proj, reference_TC).backward()
    projected_TC.requires_grad_()
    _, (_, backward_code) = run_fw_bw_and_get_code(
        lambda: loss_fn(_split_in_proj, projected_TC)
    )
    backward_kernels = _triton_kernels(backward_code)
    assert backward_kernels and all(
        name.startswith("triton_for_") for name in backward_kernels
    ), backward_kernels
    assert torch.equal(projected_TC.grad, reference_TC.grad)
