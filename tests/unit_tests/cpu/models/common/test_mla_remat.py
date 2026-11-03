# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch_remat as remat
from torch.testing._internal.common_utils import DeterministicGuard

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import InnerAttention, MLAAttention
from torchtitan.models.common.linear import Linear


class CPUAttention(InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    def __init__(self, config):
        super().__init__()

    def forward(self, q, k, v):
        scores = torch.einsum("thd,shd->hts", q, k) / q.shape[-1] ** 0.5
        return torch.einsum("hts,shd->thd", scores.softmax(dim=-1), v)


@pytest.mark.parametrize(
    "save,recompute,projection_calls,kernel_calls",
    [
        ([], [], 2, 2),
        (["attention.inner_attention"], [], 2, 1),
        (["attention.inner_attention.wkv_b.linear"], [], 1, 2),
        (["*"], [], 1, 1),
        (["*"], ["attention.inner_attention.wkv_b.linear"], 2, 1),
        (["*"], ["attention.inner_attention"], 1, 2),
    ],
)
def test_mla_remat_projection_and_kernel_are_independent(
    save, recompute, projection_calls, kernel_calls
):
    with DeterministicGuard(True), spmd.no_typecheck():
        torch.manual_seed(42)
        attention = MLAAttention.Config(
            wkv_b=Linear.Config(in_features=8, out_features=32),
            packed_kv_head_dim=16,
            inner_attention=CPUAttention.Config(),
        ).build()
        block = Module()
        block.attention = Module()
        block.attention.inner_attention = attention
        block.configure_remat_regions(save, recompute)
        inputs = [torch.randn(*shape) for shape in ((8, 2, 12), (8, 8), (8, 4))]

        def run(checkpoint):
            attention.zero_grad(set_to_none=True)
            local_inputs = [x.detach().clone().requires_grad_() for x in inputs]
            forward = remat.checkpoint()(attention) if checkpoint else attention
            output = forward(*local_inputs)
            output.square().sum().backward()
            grads = [x.grad for x in (*local_inputs, *attention.parameters())]
            assert all(grad is not None for grad in grads)
            return output.detach(), [grad.detach().clone() for grad in grads]

        expected = run(checkpoint=False)
        with (
            patch.object(
                attention.wkv_b, "_linear", wraps=attention.wkv_b._linear
            ) as linear,
            patch.object(
                attention.inner_attention,
                "forward",
                wraps=attention.inner_attention.forward,
            ) as kernel,
        ):
            actual = run(checkpoint=True)
        assert linear.call_count == projection_calls
        assert kernel.call_count == kernel_calls
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
