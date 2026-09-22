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
from torchtitan.models.common.linear import CastLinear, Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module


class _Model(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        experts: GroupedExperts.Config
        projection: Linear.Config


def _config():
    return _Model.Config(
        experts=GroupedExperts.Config(dim=64, hidden_dim=64, num_experts=2),
        projection=Linear.Config(in_features=64, out_features=32),
    )


class MXQATTransformTest(unittest.TestCase):
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
        activation_fn = config.experts.activation_fn
        transform.transform(config)
        self.assertIs(type(config.experts), config_type)
        self.assertIs(config.experts.activation_fn, activation_fn)
        self.assertIs(type(config.projection), Linear.Config)

    def test_resolved_selection_and_kernel_preference(self):
        config = _config()
        transform = MXQATTransform.from_weight_fqns(
            config,
            {
                "experts.w1_EFD",
                "experts.w2_EDF",
                "experts.w3_EFD",
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

    def test_gpt_oss_expert_layout_preserves_biases_and_config(self):
        from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts

        config = _config()
        config.experts = GptOssGroupedExperts.Config(
            dim=64, hidden_dim=64, num_experts=2, swiglu_limit=5.0
        )
        transform = MXQATTransform.from_weight_fqns(
            config, {"experts.mlp1_weight_EGD", "experts.mlp2_weight_EDF"}
        )
        transform.transform(config)
        module = config.experts.build()
        self.assertIsInstance(module, GptOssGroupedExperts)
        self.assertEqual(module.swiglu_limit, 5.0)
        self.assertEqual(
            set(module.state_dict()),
            {"mlp1_weight_EGD", "mlp1_bias_EG", "mlp2_weight_EDF", "mlp2_bias_ED"},
        )

    def test_backend_mismatch_fails_before_model_config_changes(self):
        config = _config()
        transform = MXQATTransform()
        transform.weight_fake_quant_config = replace(
            transform.weight_fake_quant_config, kernel_preference=KernelPreference.AUTO
        )
        with self.assertRaisesRegex(ValueError, "matching.*kernel_preference"):
            transform.transform(config)
        self.assertIs(type(config.experts), GroupedExperts.Config)
        # Weight-only dense QAT has no activation backend to match.
        transform.grouped_expert_fqns = ()
        transform.linear_fqns = ("projection",)
        transform.transform(config)
        self.assertTrue(type(config.projection)._owner._mx_qat)

    def test_rejects_partial_group_and_unknown_weights(self):
        for weights in ({"experts.w1_EFD"}, {"missing.weight"}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                MXQATTransform.from_weight_fqns(_config(), weights)

    def test_rejects_unknown_fqn_before_mutation(self):
        for name in ("missing", "experts"):
            with self.subTest(name=name):
                config = _config()
                with self.assertRaisesRegex(ValueError, "did not match"):
                    MXQATTransform(linear_fqns=(name,)).transform(config)
                self.assertIs(type(config.experts), GroupedExperts.Config)

    def test_rejects_duplicate_transform_sequence(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            transform_model_config_(_config(), [MXQATTransform(), MXQATTransform()])

    def test_preserves_custom_linear_contract_by_rejecting_it(self):
        config = _config()
        config.projection = CastLinear.Config(in_features=64, out_features=32)
        with self.assertRaisesRegex(ValueError, "custom forward"):
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
        MXQATTransform(grouped_expert_fqns=(), linear_fqns=("projection",)).transform(
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
