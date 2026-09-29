# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from torchao.prototype.qat import mx_fake_quantize, mx_fake_quantized_grouped_mm
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.quantization.mx_qat.experts import _get_mx_qat_grouped_linear_cls


class MXQATGroupedLinearTest(unittest.TestCase):
    def test_fused_projections_match_independent_linears_and_gradients(self) -> None:
        torch.manual_seed(17)
        cls = _get_mx_qat_grouped_linear_cls(GroupedLinear)
        config = cls.Config(
            in_features=64, out_features=32, group_size=2, num_linears=2
        )
        module = config.build().to(dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.normal_(std=0.1)
        activation = torch.randn(5, 64, dtype=torch.bfloat16, requires_grad=True)
        reference_input = activation.detach().clone().requires_grad_()
        reference_weight = module.weight.detach().clone().requires_grad_()
        offsets = torch.tensor([2, 5], dtype=torch.int32)

        actual = module(activation, offsets)
        inputs = mx_fake_quantize(reference_input, config.activation_fake_quant_config)
        weights = mx_fake_quantize(reference_weight, config.weight_fake_quant_config)
        expected = torch.cat(
            [
                torch.stack([inputs[:2] @ weights[0, p].T for p in range(2)], dim=1),
                torch.stack([inputs[2:] @ weights[1, p].T for p in range(2)], dim=1),
            ]
        )
        self.assertEqual(actual.shape, (5, 2, 32))
        torch.testing.assert_close(actual, expected)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        torch.testing.assert_close(module.weight.grad, reference_weight.grad)
        # Separate BF16 GEMMs round the input-gradient sums independently.
        torch.testing.assert_close(
            activation.grad, reference_input.grad, atol=1e-4, rtol=0.02
        )
        self.assertEqual(module.weight.grad.shape, (2, 2, 32, 64))
        for projection in range(2):
            self.assertGreater(module.weight.grad[:, projection].float().norm(), 0)

    def test_keeps_bf16_masters_and_applies_both_fake_quantizers(self) -> None:
        cls = _get_mx_qat_grouped_linear_cls(GroupedLinear)
        module = cls.Config(in_features=64, out_features=64, group_size=2).build()
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
            output = module(activation, offsets)

        output.float().sum().backward()
        self.assertEqual(grouped_mm.call_count, 1)
        self.assertIsNotNone(activation.grad)
        self.assertIsNotNone(module.weight.grad)
        self.assertTrue(torch.isfinite(activation.grad).all())
        self.assertTrue(torch.isfinite(module.weight.grad).all())
        before = module.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, module.weight))
        self.assertEqual(parameter_ids, {id(p) for p in module.parameters()})


if __name__ == "__main__":
    unittest.main()
