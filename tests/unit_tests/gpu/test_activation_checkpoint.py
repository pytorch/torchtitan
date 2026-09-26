# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch
import torch_remat as remat

from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleDict


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


class _MultipleBlockContainerModel(Module):
    def __init__(self):
        super().__init__()
        self.first_blocks = ModuleDict({"0": TransformerBlock()})
        self.second_blocks = ModuleDict({"0": TransformerBlock()})

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        return self.first_blocks["0"](x_BD) + self.second_blocks["0"](x_BD)


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


class TestActivationCheckpointing(unittest.TestCase):
    def test_custom_block_containers(self):
        model = _MultipleBlockContainerModel()
        FullAC.Config().build().apply(
            model,
            block_container_fqns=("first_blocks", "second_blocks"),
        )
        model(torch.randn(8, 32, requires_grad=True)).backward()

        for blocks in (model.first_blocks, model.second_blocks):
            block = blocks["0"]
            assert isinstance(block, TransformerBlock)
            self.assertEqual(block.input_projection.num_forwards, 2)

    def test_full_and_selective_recomputation(self):
        for policy_config, expected_counts in (
            (FullAC.Config(), (2, 2, 2)),
            (SelectiveAC.Config(), (1, 1, 1)),
        ):
            with self.subTest(policy=type(policy_config).__qualname__):
                model = ToyModel()
                policy_config.build().apply(model)
                _run_forward_backward(model, torch.randn(8, 32))

                block = model.layers["0"]
                assert isinstance(block, TransformerBlock)
                self.assertEqual(
                    (
                        block.input_projection.num_forwards,
                        block.inner_compute.num_forwards,
                        block.output_projection.num_forwards,
                    ),
                    expected_counts,
                )

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
            (lambda: FullAC.Config(preserve_rng_state=True), "preserve_rng_state"),
            (lambda: SelectiveAC.Config(debug=True), "debug option"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                config_factory()


if __name__ == "__main__":
    unittest.main()
