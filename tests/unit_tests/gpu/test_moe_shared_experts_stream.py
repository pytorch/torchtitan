# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
    make_shared_expert_ffn_config,
)

T, D, H, E, K = 4096, 512, 1024, 8, 2


class _Delayed(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        # Delaying the shared experts makes a missing stream join show up as wrong values.
        torch.cuda._sleep(100_000_000)
        return self.inner(x_TD)


class _ReadSharedExpertGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_TD, moe, grads):
        ctx.moe, ctx.grads = moe, grads
        return x_TD.view_as(x_TD)

    @staticmethod
    def backward(ctx, grad_TD):
        # FSDP reads gradients at this point: on the main stream, once the MoE input gradient exists.
        ctx.grads.update(
            {n: p.grad.clone() for n, p in ctx.moe.shared_experts.named_parameters()}
        )
        return grad_TD, None, None


def _delay(grad: torch.Tensor) -> torch.Tensor:
    torch.cuda._sleep(300_000_000)
    return grad


def _build(shared_experts_stream: bool):
    config = make_moe_config(
        num_experts=E,
        router=make_router_config(
            dim=D,
            num_experts=E,
            gate_param_init={},
            score_func=Sigmoid.Config(),
            top_k=K,
        ),
        routed_experts=make_routed_experts_config(
            dim=D, hidden_dim=H, num_experts=E, top_k=K, param_init={}
        ),
        shared_experts=make_shared_expert_ffn_config(
            dim=D, hidden_dim=H, w1_param_init={}, w2w3_param_init={}
        ),
    )
    config.shared_experts_stream = shared_experts_stream
    torch.manual_seed(0)
    moe = config.build().cuda()
    with torch.no_grad():
        for param in moe.parameters():
            param.normal_(0, 0.05)
    return moe


def _run(shared_experts_stream: bool):
    moe = _build(shared_experts_stream)
    moe.shared_experts = _Delayed(moe.shared_experts)
    x_TD = torch.randn(T, D, device="cuda", requires_grad=True)
    out_TD = moe(x_TD.clone())
    out_TD.square().sum().backward()
    torch.cuda.synchronize()
    grads = {name: param.grad for name, param in moe.named_parameters()}
    return moe, out_TD.detach(), x_TD.grad, grads


def _read_shared_expert_grads_in_backward(shared_experts_stream: bool):
    moe = _build(shared_experts_stream)
    if shared_experts_stream:
        # Delaying the side-stream accumulation makes a missing join show up as stale gradients.
        for param in moe.shared_experts.parameters():
            param.register_hook(_delay)
    torch.manual_seed(1)
    xs = [torch.randn(T, D, device="cuda", requires_grad=True) for _ in range(2)]
    grads = {}
    for x_TD in xs:
        moe(_ReadSharedExpertGrads.apply(x_TD, moe, grads)).square().sum().backward()
    torch.cuda.synchronize()
    return grads


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestSharedExpertsStream(unittest.TestCase):
    def test_side_stream_matches_the_default_stream(self):
        _, out_ref, grad_x_ref, grads_ref = _run(shared_experts_stream=False)
        moe, out, grad_x, grads = _run(shared_experts_stream=True)

        self.assertIsNotNone(moe._shared_stream)
        torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
        self.assertEqual(grads.keys(), grads_ref.keys())
        for name, grad in grads.items():
            torch.testing.assert_close(grad, grads_ref[name], rtol=0, atol=0, msg=name)
        # The shared experts now run first, so the input gradient sums its branches in another order.
        torch.testing.assert_close(grad_x, grad_x_ref, rtol=1e-5, atol=1e-4)

    def test_input_hook_orders_the_side_stream_gradient_accumulation(self):
        grads_ref = _read_shared_expert_grads_in_backward(shared_experts_stream=False)
        grads = _read_shared_expert_grads_in_backward(shared_experts_stream=True)

        self.assertTrue(grads)
        self.assertEqual(grads.keys(), grads_ref.keys())
        for name, grad in grads.items():
            torch.testing.assert_close(grad, grads_ref[name], rtol=0, atol=0, msg=name)


if __name__ == "__main__":
    unittest.main()
