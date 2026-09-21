# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from torchao.prototype.qat import mx_fake_quantized_grouped_mm
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.quantization.mx_qat.experts import _get_mx_qat_grouped_experts_cls


class MXQATGroupedExpertsTest(unittest.TestCase):
    def test_keeps_bf16_masters_and_applies_both_fake_quantizers(self) -> None:
        cls = _get_mx_qat_grouped_experts_cls(GroupedExperts)
        module = cls.Config(dim=64, hidden_dim=64, num_experts=2).build()
        module.to(dtype=torch.bfloat16)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.normal_(mean=0.0, std=0.1)

        parameter_ids = {id(parameter) for parameter in module.parameters()}
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(parameter_ids, optimizer_ids)
        self.assertTrue(
            all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())
        )
        self.assertFalse(
            any(
                key.endswith(("weight_packed", "weight_scale"))
                for key in module.state_dict()
            )
        )

        activation = torch.randn(
            4,
            64,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        offsets = torch.tensor([2, 4], dtype=torch.int32)
        with patch(
            "torchao.prototype.qat.mx_fake_quantized_grouped_mm",
            wraps=mx_fake_quantized_grouped_mm,
        ) as grouped_mm:
            output = module._grouped_mm(
                A=activation,
                weight_EOI=module.w1_EFD,
                offs=offsets,
            )

        output.float().sum().backward()
        self.assertEqual(grouped_mm.call_count, 1)
        self.assertIsNotNone(activation.grad)
        self.assertIsNotNone(module.w1_EFD.grad)
        self.assertTrue(torch.isfinite(activation.grad).all())
        self.assertTrue(torch.isfinite(module.w1_EFD.grad).all())
        before = module.w1_EFD.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, module.w1_EFD))
        self.assertEqual(parameter_ids, {id(p) for p in module.parameters()})


if __name__ == "__main__":
    unittest.main()
