# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.attention import gate_attention_output

# T tokens, H heads, V value dims
T, H, V = 16, 4, 8


@pytest.fixture(autouse=True)
def reset_local_compile():
    apply_local_compile([])
    torch._dynamo.reset()
    yield
    apply_local_compile([])


@pytest.mark.parametrize("gate_layout", ["wq_half", "own_linear"])
@pytest.mark.parametrize("output_layout", ["tokens_first", "heads_first"])
def test_gate_attention_output_matches_inline_reference(
    gate_layout: str, output_layout: str
) -> None:
    """Eager: each model's former inline gating, bit for bit. Compiled: within a few percent.

    ``wq_half``: Qwen3.5's gate, the second half of each head of a 2x-wide wq output.
    ``own_linear``: Muse Glimmer's gate, its own ``[T, H * V]`` projection.
    ``heads_first``: the attention output is a ``[H, T, V]`` buffer seen as ``[T, H, V]``.
    """
    torch.manual_seed(0)
    output_shape = (T, H, V) if output_layout == "tokens_first" else (H, T, V)
    gate_shape = (T, H, 2 * V) if gate_layout == "wq_half" else (T, H * V)
    output_base = torch.randn(output_shape, dtype=torch.bfloat16)
    gate_base = torch.randn(gate_shape, dtype=torch.bfloat16)
    grad = torch.randn(T, H * V, dtype=torch.bfloat16)

    def run(gated_output):
        output_leaf = output_base.clone().requires_grad_()
        gate_leaf = gate_base.clone().requires_grad_()
        output = (
            output_leaf
            if output_layout == "tokens_first"
            else output_leaf.transpose(0, 1)
        )
        gated = gated_output(output, gate_leaf)
        gated.backward(grad)
        return gated, output_leaf.grad, gate_leaf.grad

    def inline_reference(output, gate_leaf):
        if gate_layout == "wq_half":
            gate = gate_leaf.chunk(2, dim=-1)[1]
            return (output.contiguous() * torch.sigmoid(gate)).view(T, -1)
        return output.contiguous().view(T, -1) * torch.sigmoid(gate_leaf)

    def shared_function(output, gate_leaf):
        if gate_layout == "wq_half":
            return gate_attention_output(output, gate_leaf.chunk(2, dim=-1)[1])
        return gate_attention_output(output, gate_leaf.view(T, H, V))

    expected = run(inline_reference)
    for actual, ref in zip(run(shared_function), expected, strict=True):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)

    apply_local_compile(["attention_output_gate"])
    # The region keeps the sigmoid in fp32. Eager's backward computes 1 - sigmoid from a bf16 sigmoid, which
    # loses bits when the sigmoid is near 1, so the gate grads differ by a few percent there.
    for actual, ref in zip(run(shared_function), expected, strict=True):
        torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-3)
