# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.activation import ClampedSwiGLU, SiTUGLU, SwiGLU


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSwiGLULocalCompile(unittest.TestCase):
    def setUp(self):
        apply_local_compile(["fused_binary_activation"])

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_forward_and_backward_emit_triton(self):
        from torch._inductor.utils import run_fw_bw_and_get_code

        swiglu = SwiGLU.Config().build()
        gate = torch.randn(
            2048,
            1024,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.randn_like(gate, requires_grad=True)

        _, codes = run_fw_bw_and_get_code(
            lambda: swiglu(torch.stack([gate, up], dim=-2))
        )

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)
        self.assertTrue(any("sigmoid" in code for code in codes))

    def test_unbinds_inside_compiled_region(self):
        for activation_fn in (
            SwiGLU.Config().build(),
            ClampedSwiGLU.Config().build(),
            SiTUGLU.Config(beta=4.0, linear_beta=25.0).build(),
        ):
            with self.subTest(type(activation_fn).__name__):
                gate_up = torch.randn(
                    64, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                output = activation_fn(gate_up)
                # One edge, straight to gate_up. An eager unbind would give two
                # UnbindBackward0 edges here: the node that stacks the two gradients.
                next_functions = output.grad_fn.next_functions
                self.assertEqual(
                    [type(fn).__name__ for fn, _ in next_functions], ["AccumulateGrad"]
                )

    def test_keeps_f_static_and_ignores_offsets(self):
        from torch._dynamo.utils import counters

        swiglu = SwiGLU.Config().build()
        offsets = torch.tensor([8, 16], device="cuda", dtype=torch.int32)
        counters.clear()
        for num_tokens, hidden_dim, kwargs in (
            (64, 128, {}),
            (32, 256, {"offsets": offsets}),
            (48, 128, {"offsets": offsets}),
            (40, 256, {}),
        ):
            gate_up = torch.randn(
                num_tokens,
                2,
                hidden_dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            swiglu(gate_up, **kwargs).sum().backward()
        # 3 graphs: the first call (all static), then one any-T graph per F. A dynamic F,
        # or reading offsets in the compiled region, would change the count.
        self.assertEqual(counters["stats"]["unique_graphs"], 3)

    def test_forward_and_backward_match_reference(self):
        swiglu = SwiGLU.Config().build()
        gate = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.randn_like(gate, requires_grad=True)
        grad_output = torch.randn_like(gate)

        output = swiglu(torch.stack([gate, up], dim=-2))
        grad_gate, grad_up = torch.autograd.grad(
            output,
            (gate, up),
            grad_output,
        )

        reference_gate = gate.detach().clone().requires_grad_()
        reference_up = up.detach().clone().requires_grad_()
        reference_output = F.silu(reference_gate) * reference_up
        reference_grad_gate, reference_grad_up = torch.autograd.grad(
            reference_output,
            (reference_gate, reference_up),
            grad_output,
        )

        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(grad_gate, reference_grad_gate)
        torch.testing.assert_close(grad_up, reference_grad_up)

    def test_gpt_oss_forward_and_backward_are_batch_invariant(self):
        swiglu = ClampedSwiGLU.Config().build()
        gate = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.randn_like(gate, requires_grad=True)
        grad_output = torch.randn_like(gate)

        output = swiglu(torch.stack([gate, up], dim=-2))
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
            output_part = swiglu(torch.stack([gate_part, up_part], dim=-2))
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

    def test_gpt_oss_forward_and_backward_match_reference(self):
        limit = 2.0
        swiglu = ClampedSwiGLU.Config(swiglu_limit=limit).build()
        gate = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.randn_like(gate, requires_grad=True)
        grad_output = torch.randn_like(gate)

        output = swiglu(torch.stack([gate, up], dim=-2))
        grad_gate, grad_up = torch.autograd.grad(
            output,
            (gate, up),
            grad_output,
        )

        reference_gate = gate.detach().clone().requires_grad_()
        reference_up = up.detach().clone().requires_grad_()
        clamped_gate = reference_gate.clamp(max=limit)
        clamped_up = reference_up.clamp(min=-limit, max=limit)
        silu = clamped_gate * torch.sigmoid(1.702 * clamped_gate)
        reference_output = torch.addcmul(silu, silu, clamped_up)
        reference_grad_gate, reference_grad_up = torch.autograd.grad(
            reference_output,
            (reference_gate, reference_up),
            grad_output,
        )

        torch.testing.assert_close(output, reference_output, rtol=2e-2, atol=1e-2)
        torch.testing.assert_close(grad_gate, reference_grad_gate, rtol=2e-2, atol=1e-2)
        torch.testing.assert_close(grad_up, reference_grad_up, rtol=2e-2, atol=1e-2)

    def test_grouped_offsets_match_reference(self):
        swiglu = SwiGLU.Config().build()
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

        output = swiglu(torch.stack([gate, up], dim=-2), offsets=offsets)
        grad_gate, grad_up = torch.autograd.grad(
            output,
            (gate, up),
            grad_output,
        )

        reference_gate = gate.detach().clone().requires_grad_()
        reference_up = up.detach().clone().requires_grad_()
        reference_output = F.silu(reference_gate) * reference_up
        reference_grad_gate, reference_grad_up = torch.autograd.grad(
            reference_output,
            (reference_gate, reference_up),
            grad_output,
        )

        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(grad_gate, reference_grad_gate)
        torch.testing.assert_close(grad_up, reference_grad_up)

    def test_forward_and_backward_are_batch_invariant(self):
        swiglu = SwiGLU.Config().build()
        gate = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.randn_like(gate, requires_grad=True)
        grad_output = torch.randn_like(gate)

        output = swiglu(torch.stack([gate, up], dim=-2))
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
            output_part = swiglu(torch.stack([gate_part, up_part], dim=-2))
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
