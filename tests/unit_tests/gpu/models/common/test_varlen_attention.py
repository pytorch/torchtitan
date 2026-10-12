# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.models.common.attention import (
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _metadata():
    boundaries = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    return VarlenAttentionMetadata(boundaries, boundaries, 32, 32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("upcast_qk", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_forward_backward_precision(dtype, upcast_qk, causal):
    torch.manual_seed(42)
    attention = VarlenInnerAttention.Config(
        window_size=(-1, 0) if causal else (-1, -1)
    ).build()
    inputs = [
        torch.randn(
            64,
            4,
            64,
            device="cuda",
            dtype=torch.float32 if upcast_qk and index < 2 else dtype,
            requires_grad=True,
        )
        for index in range(3)
    ]
    reference_inputs = [x.detach().double().requires_grad_() for x in inputs]
    actual = attention(*inputs, attention_metadata=_metadata())
    reference_q, reference_k, reference_v = (
        x.reshape(2, 32, 4, 64).transpose(1, 2) for x in reference_inputs
    )
    with sdpa_kernel(SDPBackend.MATH):
        expected = (
            F.scaled_dot_product_attention(
                reference_q,
                reference_k,
                reference_v,
                is_causal=causal,
            )
            .transpose(1, 2)
            .reshape(64, 4, 64)
        )
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    expected_grads = torch.autograd.grad(
        expected, reference_inputs, grad_output.double()
    )

    assert actual.dtype == inputs[0].dtype
    tolerance = 0.001 if dtype == torch.float16 else 0.01
    for result, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        relative_error = (result.double() - reference).norm() / reference.norm()
        assert relative_error.item() < tolerance


def test_fp32_fallback_matches_legacy_forward_backward():
    torch.manual_seed(42)
    attention = VarlenInnerAttention.Config().build()
    inputs = [
        torch.randn(64, 4, 64, device="cuda", requires_grad=True) for _ in range(3)
    ]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    metadata = _metadata()
    actual = attention(*inputs, attention_metadata=metadata)
    expected = attention(
        *(x.bfloat16() for x in reference_inputs), attention_metadata=metadata
    ).float()
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    torch.testing.assert_close(
        (actual, actual_grads), (expected, expected_grads), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["compile", "trace"])
def test_compiled_and_traced_steps_match_eager(dtype, mode):
    torch.manual_seed(42)
    attention = VarlenInnerAttention.Config().build()
    inputs = [
        torch.randn(64, 4, 64, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    metadata = _metadata()

    def step(q, k, v, *, attention_metadata):
        output = attention(q, k, v, attention_metadata=attention_metadata)
        loss = output.float().square().mean()
        return output, loss, torch.autograd.grad(loss, (q, k, v))

    expected = step(*inputs, attention_metadata=metadata)
    if mode == "trace":
        traced = minimal_fx_tracer(step, module=attention)(
            *inputs, attention_metadata=metadata
        )
        actual = run_traced(traced, module=attention)(
            *inputs, attention_metadata=metadata
        )
    else:
        compiled = torch.compile(attention, fullgraph=True)
        output = compiled(*inputs, attention_metadata=metadata)
        loss = output.float().square().mean()
        actual = output, loss, torch.autograd.grad(loss, inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
