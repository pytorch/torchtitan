# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.activation import SiTUGLU


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSiTUGLULocalCompile(unittest.TestCase):
    def setUp(self):
        apply_local_compile(["situglu"])

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_forward_and_backward_emit_triton(self):
        from torch._inductor.utils import run_fw_bw_and_get_code

        situglu = SiTUGLU.Config(beta=4.0, linear_beta=25.0).build()
        gate = torch.randn(
            2048,
            1024,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.randn_like(gate, requires_grad=True)

        _, codes = run_fw_bw_and_get_code(lambda: situglu(gate, up))

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)

    def test_forward_and_backward_match_reference_with_offsets(self):
        beta = 4.0
        linear_beta = 25.0
        situglu = SiTUGLU.Config(beta=beta, linear_beta=linear_beta).build()
        gate = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.randn_like(gate, requires_grad=True)
        offsets = torch.tensor([8, 24, 40, 48], device="cuda", dtype=torch.int32)
        grad_output = torch.randn_like(gate)

        output = situglu(gate, up, offsets=offsets)
        grad_gate, grad_up = torch.autograd.grad(
            output,
            (gate, up),
            grad_output,
        )

        reference_gate = gate.detach().clone().requires_grad_()
        reference_up = up.detach().clone().requires_grad_()
        gate_fp32 = reference_gate.float()
        up_fp32 = reference_up.float()
        gate_fp32 = beta * torch.tanh(gate_fp32 / beta) * torch.sigmoid(gate_fp32)
        up_fp32 = linear_beta * torch.tanh(up_fp32 / linear_beta)
        reference_output = (gate_fp32 * up_fp32).to(reference_gate.dtype)
        reference_grad_gate, reference_grad_up = torch.autograd.grad(
            reference_output,
            (reference_gate, reference_up),
            grad_output,
        )

        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(grad_gate, reference_grad_gate)
        torch.testing.assert_close(grad_up, reference_grad_up)

    def test_forward_and_backward_are_batch_invariant(self):
        situglu = SiTUGLU.Config(beta=4.0, linear_beta=25.0).build()
        gate = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.randn_like(gate, requires_grad=True)
        grad_output = torch.randn_like(gate)

        output = situglu(gate, up)
        grad_gate, grad_up = torch.autograd.grad(
            output,
            (gate, up),
            grad_output,
        )

        split_outputs = []
        split_grad_gates = []
        split_grad_ups = []
        for gate_part, up_part, grad_part in zip(
            gate.detach().chunk(2),
            up.detach().chunk(2),
            grad_output.chunk(2),
            strict=True,
        ):
            gate_part.requires_grad_()
            up_part.requires_grad_()
            output_part = situglu(gate_part, up_part)
            grad_gate_part, grad_up_part = torch.autograd.grad(
                output_part,
                (gate_part, up_part),
                grad_part,
            )
            split_outputs.append(output_part)
            split_grad_gates.append(grad_gate_part)
            split_grad_ups.append(grad_up_part)

        self.assertTrue(torch.equal(output, torch.cat(split_outputs)))
        self.assertTrue(torch.equal(grad_gate, torch.cat(split_grad_gates)))
        self.assertTrue(torch.equal(grad_up, torch.cat(split_grad_ups)))


if __name__ == "__main__":
    unittest.main()
