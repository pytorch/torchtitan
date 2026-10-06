# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch
import torch_remat as remat

from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleDict


_effectful_call_count = 0


@torch.library.custom_op("torchtitan_test::effectful_identity", mutates_args=())
def _effectful_identity(x: torch.Tensor) -> torch.Tensor:
    """Return ``x`` through an ordered operation and count its executions."""
    global _effectful_call_count
    _effectful_call_count += 1
    return x.clone()


@_effectful_identity.register_fake
def _effectful_identity_fake(x: torch.Tensor) -> torch.Tensor:
    """Describe the ordered operation's output during fake execution."""
    return torch.empty_like(x)


def _effectful_identity_backward(
    _ctx: object, grad_output: torch.Tensor
) -> torch.Tensor:
    """Propagate gradients through the identity operation."""
    return grad_output


_effectful_identity.register_autograd(_effectful_identity_backward)
_effectful_identity.register_effect(torch.library.EffectType.ORDERED)


class _CountingLinear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        ).build()
        self.num_forwards = 0

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        self.num_forwards += 1
        return self.linear(x_BD)


class TransformerBlock(Module):
    def __init__(self):
        super().__init__()
        self.input_projection = _CountingLinear(32, 32)
        self.inner_compute = _CountingLinear(32, 32)
        self.output_projection = _CountingLinear(32, 32)

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        hidden_BD = remat.region(
            self.input_projection,
            self.remat_region_name("input_projection"),
            recompute=self.remat_should_recompute("input_projection"),
        )(x_BD)
        hidden_BD = remat.region(
            self.inner_compute,
            self.remat_region_name("inner_compute"),
            recompute=self.remat_should_recompute("inner_compute"),
        )(hidden_BD)
        output_BD = remat.region(
            self.output_projection,
            self.remat_region_name("output_projection"),
            recompute=self.remat_should_recompute("output_projection"),
        )(hidden_BD)
        remat.recompute_needs_tensor(output_BD)
        return output_BD.sum()


class ToyModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({"0": TransformerBlock()})

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_BD)


def _run_forward_backward(
    model: ToyModel,
    x_BD: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    input_BD = x_BD.detach().clone().requires_grad_(True)
    output = model(input_BD)
    output.backward()
    assert input_BD.grad is not None
    parameter_grads = [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return output.detach(), input_BD.grad.detach().clone(), parameter_grads


def _unwrap_transformer_block(module: Module) -> TransformerBlock:
    if isinstance(module, TransformerBlock):
        return module
    block = module.get_submodule("_checkpoint_wrapped_module")
    assert isinstance(block, TransformerBlock)
    return block


class TestActivationCheckpointing(unittest.TestCase):
    def test_full_ac_does_not_recompute_registered_effects(self):
        """FullAC must save, rather than replay, registered ordered effects."""

        class EffectfulBlock(Module):
            def forward(self, x):
                return _effectful_identity(x).sin()

        class EffectfulModel(Module):
            def __init__(self):
                super().__init__()
                self.layers = ModuleDict({"0": EffectfulBlock()})

            def forward(self, x):
                return self.layers["0"](x)

        global _effectful_call_count
        _effectful_call_count = 0
        model = EffectfulModel()
        FullAC.Config().build().apply(model)

        for iteration in range(2):
            x = torch.randn(8, requires_grad=True)
            output = model(x).sum()
            output.backward()
            torch.testing.assert_close(output, x.sin().sum())
            torch.testing.assert_close(x.grad, x.cos())
            self.assertEqual(_effectful_call_count, iteration + 1)

    def test_full_and_selective_recomputation(self):
        for policy_config, expected_counts in (
            (FullAC.Config(), (2, 2, 2)),
            (SelectiveAC.Config(), (1, 1, 1)),
        ):
            with self.subTest(policy=type(policy_config).__qualname__):
                model = ToyModel()
                policy_config.build().apply(model)
                _run_forward_backward(model, torch.randn(8, 32))

                block = _unwrap_transformer_block(model.layers["0"])
                self.assertEqual(
                    (
                        block.input_projection.num_forwards,
                        block.inner_compute.num_forwards,
                        block.output_projection.num_forwards,
                    ),
                    expected_counts,
                )

    def test_selective_recomputes_routed_experts(self):
        class RegionLinear(Module):
            def __init__(self):
                super().__init__()
                self.projection = _CountingLinear(32, 32)

            def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
                return remat.region(
                    self.projection,
                    self.remat_region_name("grouped_mm"),
                    recompute=self.remat_should_recompute("grouped_mm"),
                )(x_BD)

        class Experts(Module):
            def __init__(self):
                super().__init__()
                self.w13 = RegionLinear()
                self.w2 = RegionLinear()

            def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
                hidden_BD = self.w13(x_BD)
                remat.recompute_needs_tensor(hidden_BD)
                return self.w2(hidden_BD.relu())

        class ExpertsBlock(Module):
            def __init__(self):
                super().__init__()
                self.routed_experts = Experts()
                self.shared_experts = RegionLinear()

            def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
                routed_BD = self.routed_experts(x_BD)
                shared_BD = self.shared_experts(x_BD)
                remat.recompute_needs_tensor(routed_BD, shared_BD)
                output_BD = routed_BD + shared_BD
                remat.recompute_needs_tensor(output_BD)
                return output_BD.sum()

        model = Module()
        model.layers = ModuleDict({"0": ExpertsBlock()})
        SelectiveAC.Config().build().apply(model)
        block = model.layers["0"]
        block(torch.randn(8, 32, requires_grad=True)).backward()

        # Only the routed-expert w13 replays; w2 and the shared expert are saved.
        self.assertEqual(block.routed_experts.w13.projection.num_forwards, 2)
        self.assertEqual(block.routed_experts.w2.projection.num_forwards, 1)
        self.assertEqual(block.shared_experts.projection.num_forwards, 1)

    def test_full_and_selective_match_uncheckpointed_model(self):
        torch.manual_seed(42)
        baseline = ToyModel()
        x_BD = torch.randn(8, 32)
        expected = _run_forward_backward(baseline, x_BD)

        for policy_config in (FullAC.Config(), SelectiveAC.Config()):
            with self.subTest(policy=type(policy_config).__qualname__):
                model = deepcopy(baseline)
                policy_config.build().apply(model)
                actual = _run_forward_backward(model, x_BD)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                self.assertEqual(len(actual[2]), len(expected[2]))
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad,
                        expected_grad,
                        rtol=0,
                        atol=0,
                    )

    def test_remat_policies_reject_unsupported_options(self):
        for config_factory, message in (
            (
                lambda: SelectiveAC.Config(preserve_rng_state=True),
                "preserve_rng_state",
            ),
            (lambda: SelectiveAC.Config(debug=True), "debug option"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                config_factory()

    def test_selective_ac_policy_is_fixed(self):
        for kwargs in (
            {"save_regions": ["attention.*"]},
            {"recompute_regions": []},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "Use RegionAC"
            ):
                SelectiveAC.Config(**kwargs)

    def test_recompute_regions_override_save_regions(self):
        model = ToyModel()
        RegionAC.Config(
            save_regions=["*"], recompute_regions=["inner_compute"]
        ).build().apply(model)
        _run_forward_backward(model, torch.randn(8, 32))

        block = _unwrap_transformer_block(model.layers["0"])
        self.assertEqual(
            (
                block.input_projection.num_forwards,
                block.inner_compute.num_forwards,
                block.output_projection.num_forwards,
            ),
            (1, 2, 1),
        )


if __name__ == "__main__":
    unittest.main()
