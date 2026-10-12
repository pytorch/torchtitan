# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.qwen3_5.moe import sigmoid_gate


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSigmoidGateLocalCompile(unittest.TestCase):
    def setUp(self):
        apply_local_compile(["shared_expert_gate"])
        # A local generator leaves the global RNG stream unchanged for later tests.
        generator = torch.Generator(device="cuda").manual_seed(0)

        def randn(*shape: int) -> torch.Tensor:
            return torch.randn(
                *shape, device="cuda", dtype=torch.bfloat16, generator=generator
            )

        self.gate = randn(64, 1).requires_grad_()
        self.out = randn(64, 128).requires_grad_()
        self.grad_output = randn(64, 128)

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_forward_and_backward_match_reference(self):
        output = sigmoid_gate(self.gate, self.out)
        grads = torch.autograd.grad(output, (self.gate, self.out), self.grad_output)

        # Like the compiled region, keep the sigmoid in fp32 (eager rounds it to bf16).
        reference = (torch.sigmoid(self.gate.float()) * self.out.float()).bfloat16()
        reference_grads = torch.autograd.grad(
            reference, (self.gate, self.out), self.grad_output
        )

        torch.testing.assert_close(output, reference)
        for grad, reference_grad in zip(grads, reference_grads, strict=True):
            torch.testing.assert_close(grad, reference_grad)

    def test_forward_and_backward_are_batch_invariant(self):
        output = sigmoid_gate(self.gate, self.out)
        grads = torch.autograd.grad(output, (self.gate, self.out), self.grad_output)

        split_results = []
        for gate, out, grad_output in zip(
            self.gate.detach().chunk(2),
            self.out.detach().chunk(2),
            self.grad_output.chunk(2),
            strict=True,
        ):
            gate.requires_grad_()
            out.requires_grad_()
            split_output = sigmoid_gate(gate, out)
            split_grads = torch.autograd.grad(split_output, (gate, out), grad_output)
            split_results.append((split_output, *split_grads))

        for full, splits in zip((output, *grads), zip(*split_results), strict=True):
            self.assertTrue(torch.equal(full, torch.cat(splits)))


if __name__ == "__main__":
    unittest.main()
