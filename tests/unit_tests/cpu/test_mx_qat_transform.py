# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, replace

import torch
from torchao.quantization.quantize_.common import KernelPreference
from torchtitan.config.transform import MXQATTransform, transform_model_config_
from torchtitan.models.common.linear import CastLinear, GroupedLinear, Linear
from torchtitan.protocols.module import Module


class _Model(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        experts: GroupedLinear.Config
        projection: Linear.Config


def _config():
    return _Model.Config(
        experts=GroupedLinear.Config(in_features=64, out_features=64, group_size=2),
        projection=Linear.Config(in_features=64, out_features=32),
    )


class MXQATTransformTest(unittest.TestCase):
    def test_stacked_linear_preserves_forward_shape_and_gradients(self):
        from torchao.prototype.qat import mx_fake_quantize

        config = _config()
        config.projection = Linear.Config(
            in_features=64, out_features=32, num_linears=2, bias=True
        )
        transform = MXQATTransform(grouped_linear_fqns=(), linear_fqns=("projection",))
        transform.transform(config)
        module = config.projection.build()
        x = torch.randn(3, 64, requires_grad=True)
        expected_weight = module.weight.detach().clone().requires_grad_()
        expected_bias = module.bias.detach().clone().requires_grad_()
        expected_x = x.detach().clone().requires_grad_()
        expected = torch.nn.functional.linear(
            expected_x,
            mx_fake_quantize(
                expected_weight, transform.weight_fake_quant_config
            ).flatten(0, 1),
            expected_bias.flatten(),
        ).unflatten(-1, (2, 32))
        actual = module(x)
        self.assertEqual(actual.shape, (3, 2, 32))
        self.assertIs(type(module).forward, Linear.forward)
        torch.testing.assert_close(actual, expected)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        for actual_grad, expected_grad in (
            (module.weight.grad, expected_weight.grad),
            (module.bias.grad, expected_bias.grad),
            (x.grad, expected_x.grad),
        ):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_configuration_types_are_resolvable_for_cli(self):
        from typing import get_type_hints

        from torchao.prototype.qat import MXFakeQuantizeConfig

        self.assertIs(
            get_type_hints(MXQATTransform)["weight_fake_quant_config"],
            MXFakeQuantizeConfig,
        )
        config = MXQATTransform().transform(_config())
        self.assertIs(
            get_type_hints(type(config.experts))["activation_fake_quant_config"],
            MXFakeQuantizeConfig,
        )

    def test_idempotence_preserves_parent_config(self):
        config = _config()
        transform = MXQATTransform()
        transform.transform(config)
        config_type = type(config.experts)
        param_init = config.experts.param_init
        transform.transform(config)
        self.assertIs(type(config.experts), config_type)
        self.assertIs(config.experts.param_init, param_init)
        self.assertIs(type(config.projection), Linear.Config)

    def test_resolved_selection_and_kernel_preference(self):
        config = _config()
        transform = MXQATTransform.from_weight_fqns(
            config,
            {
                "experts.weight",
                "projection.weight",
            },
        )
        transform.weight_fake_quant_config = replace(
            transform.weight_fake_quant_config, kernel_preference=KernelPreference.AUTO
        )
        transform.activation_fake_quant_config = replace(
            transform.activation_fake_quant_config,
            kernel_preference=KernelPreference.AUTO,
        )
        transform.transform(config)
        self.assertEqual(
            config.experts.weight_fake_quant_config.kernel_preference,
            KernelPreference.AUTO,
        )
        self.assertEqual(
            config.experts.activation_fake_quant_config.kernel_preference,
            KernelPreference.AUTO,
        )
        self.assertEqual(
            config.projection.weight_fake_quant_config.kernel_preference,
            KernelPreference.AUTO,
        )

    def test_stacked_grouped_linear_preserves_storage_and_forward(self):
        config = _config()
        config.experts.num_linears = 2
        transform = MXQATTransform.from_weight_fqns(config, {"experts.weight"})
        transform.transform(config)
        module = config.experts.build()
        self.assertIs(type(module).forward, GroupedLinear.forward)
        self.assertEqual(module.weight.shape, (2, 2, 64, 64))
        self.assertEqual(set(module.state_dict()), {"weight"})

    def test_backend_mismatch_fails_before_model_config_changes(self):
        config = _config()
        transform = MXQATTransform()
        transform.weight_fake_quant_config = replace(
            transform.weight_fake_quant_config, kernel_preference=KernelPreference.AUTO
        )
        with self.assertRaisesRegex(ValueError, "matching.*kernel_preference"):
            transform.transform(config)
        self.assertIs(type(config.experts), GroupedLinear.Config)
        # Weight-only dense QAT has no activation backend to match.
        transform.grouped_linear_fqns = ()
        transform.linear_fqns = ("projection",)
        transform.transform(config)
        self.assertTrue(type(config.projection)._owner._mx_qat)

    def test_rejects_unknown_and_legacy_weights(self):
        for weights in ({"experts.w1_EFD"}, {"missing.weight"}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                MXQATTransform.from_weight_fqns(_config(), weights)

    def test_rejects_unknown_fqn_before_mutation(self):
        for name in ("missing", "experts"):
            with self.subTest(name=name):
                config = _config()
                with self.assertRaisesRegex(ValueError, "did not match"):
                    MXQATTransform(linear_fqns=(name,)).transform(config)
                self.assertIs(type(config.experts), GroupedLinear.Config)

    def test_rejects_duplicate_transform_sequence(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            transform_model_config_(_config(), [MXQATTransform(), MXQATTransform()])

    def test_preserves_custom_linear_contract_by_rejecting_it(self):
        config = _config()
        config.projection = CastLinear.Config(in_features=64, out_features=32)
        with self.assertRaisesRegex(ValueError, "custom linear compute"):
            MXQATTransform(linear_fqns=("projection",)).transform(config)

    def test_lora_wraps_qat_and_preserves_its_forward(self):
        from torchao.prototype.qat import mx_fake_quantize
        from torchtitan.config.transform import LinearLoRAHandler, LoRATransform

        config = transform_model_config_(
            _config(),
            [
                LoRATransform(handlers=(LinearLoRAHandler(),), rank=4),
                MXQATTransform(linear_fqns=("projection",)),
            ],
        )
        module = config.projection.build()
        with torch.no_grad():
            module.weight.normal_()
            module.lora_a.weight.normal_()
            module.lora_b.weight.zero_()
        value = torch.randn(2, 64, requires_grad=True)
        expected = torch.nn.functional.linear(
            value,
            mx_fake_quantize(module.weight, config.projection.weight_fake_quant_config),
        )
        actual = module(value)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsNone(module.weight.grad)
        self.assertGreater(torch.count_nonzero(module.lora_b.weight.grad), 0)
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_dense_qat_keeps_parameter_names_and_optimizer_identity(self):
        config = _config()
        MXQATTransform(grouped_linear_fqns=(), linear_fqns=("projection",)).transform(
            config
        )
        module = config.projection.build().to(dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.normal_()
        parameter = module.weight
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        module(torch.randn(2, 64, dtype=torch.bfloat16)).float().sum().backward()
        before = parameter.detach().clone()
        optimizer.step()
        self.assertIs(module.weight, parameter)
        self.assertEqual(set(module.state_dict()), {"weight"})
        self.assertFalse(torch.equal(before, parameter))


if __name__ == "__main__":
    unittest.main()
