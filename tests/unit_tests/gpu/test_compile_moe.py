# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.linear import GroupedLinear
from torchtitan.tools.utils import device_module, device_type


device = torch.device(device_type)


class TestGroupedLinearCompile(unittest.TestCase):
    @unittest.skipUnless(device_module.is_available(), "requires an accelerator")
    def test_grouped_linear_compile_and_numerics(self):
        """Compiled structured grouped GEMM matches independent expert GEMMs."""
        num_experts = 8
        dim = 128
        hidden_dim = 256
        w13 = (
            GroupedLinear.Config(
                group_size=num_experts,
                in_features=dim,
                out_features=hidden_dim,
                num_linears=2,
            )
            .build()
            .to(device=device)
        )
        num_tokens_per_expert = torch.tensor(
            [10, 0, 12, 9, 11, 7, 10, 13],
            dtype=torch.int32,
            device=device,
        )
        total_tokens = num_tokens_per_expert.sum().item()
        input_RI = torch.randn(
            total_tokens,
            dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        reference_input_RI = input_RI.detach().clone().requires_grad_()
        with torch.no_grad():
            w13.weight.normal_()
        reference_weight_E2OI = w13.weight.detach().clone().requires_grad_()
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        output_R2O = torch.compile(w13, fullgraph=True)(input_RI, offsets)
        reference_chunks = []
        start = 0
        for expert, count in enumerate(num_tokens_per_expert.tolist()):
            end = start + count
            reference_chunks.append(
                reference_input_RI[start:end]
                @ reference_weight_E2OI[expert].bfloat16().flatten(0, 1).T
            )
            start = end
        reference_R2O = torch.cat(reference_chunks).unflatten(-1, (2, hidden_dim))

        torch.testing.assert_close(output_R2O, reference_R2O, rtol=0, atol=0)
        grad_R2O = torch.randn_like(output_R2O)
        output_R2O.backward(grad_R2O)
        reference_R2O.backward(grad_R2O)
        torch.testing.assert_close(
            input_RI.grad,
            reference_input_RI.grad,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            w13.weight.grad,
            reference_weight_E2OI.grad,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
