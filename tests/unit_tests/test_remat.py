# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import pytest
import torch

from torchtitan.distributed.activation_checkpoint import (
    RematAC,
    validate_activation_checkpointing_compile,
)
from torchtitan.models.common.remat import (
    apply_region_selection,
    declared_regions,
    maybe_remat_recompute_needs,
    maybe_remat_save_region,
    require_torch_remat,
)
from torchtitan.protocols.module import Module, ModuleDict


class _RegionOwner(Module):
    REMAT_REGIONS = ("w1", "w2")


class _InheritedRegionOwner(_RegionOwner):
    pass


class _OverriddenRegionOwner(_RegionOwner):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _RegionTree(Module):
    def __init__(self):
        super().__init__()
        self.left = _RegionOwner()
        self.right = _RegionOwner()


class _CountingLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(4, 4, bias=False)
        self.num_forwards = 0

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        self.num_forwards += 1
        return super().forward(x_BD)


class _RematBlock(Module):
    REMAT_REGIONS = ("saved",)

    def __init__(self):
        super().__init__()
        self.saved = _CountingLinear()
        self.recomputed = _CountingLinear()

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        saved_BD = maybe_remat_save_region(self.saved, "saved", owner=self)(x_BD)
        maybe_remat_recompute_needs(self, saved_BD)
        return self.recomputed(torch.sin(saved_BD)).sum()


class _RematModel(Module):
    def __init__(self, block: Module | None = None):
        super().__init__()
        self.layers = ModuleDict({"0": block or _RematBlock()})

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_BD)


def _run_forward_backward(
    model: Module, x_BD: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    input_BD = x_BD.detach().clone().requires_grad_(True)
    output = model(input_BD)
    output.backward()
    assert input_BD.grad is not None
    parameter_grads = []
    for parameter in model.parameters():
        assert parameter.grad is not None
        parameter_grads.append(parameter.grad.detach().clone())
    return output.detach(), input_BD.grad.detach().clone(), parameter_grads


class TestRematRegions(unittest.TestCase):
    def test_region_discovery_and_glob_selection(self):
        tree = _RegionTree()

        self.assertEqual(
            declared_regions(tree),
            ["left.w1", "left.w2", "right.w1", "right.w2"],
        )
        selected, available = apply_region_selection(tree, ["*.w2", "left.w1"])

        self.assertEqual(
            selected,
            ["left.w1", "left.w2", "right.w2"],
        )
        self.assertEqual(available, declared_regions(tree))

    def test_forward_override_must_redeclare_regions(self):
        self.assertEqual(declared_regions(_InheritedRegionOwner()), ["w1", "w2"])
        self.assertEqual(declared_regions(_OverriddenRegionOwner()), [])

    def test_compile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support torch.compile"):
            validate_activation_checkpointing_compile(
                RematAC.Config(), model_compile_enabled=True
            )

        validate_activation_checkpointing_compile(
            RematAC.Config(), model_compile_enabled=False
        )

    def test_unmatched_region_pattern_is_rejected(self):
        pytest.importorskip("torch_remat")
        require_torch_remat()
        model = _RematModel()

        with self.assertRaisesRegex(ValueError, "matched no declared region"):
            RematAC.Config(save_regions=["missing"]).build().apply(model)

    def test_model_without_declared_regions_is_rejected(self):
        pytest.importorskip("torch_remat")
        require_torch_remat()
        model = _RematModel(Module())

        with self.assertRaisesRegex(ValueError, "does not declare any"):
            RematAC.Config(save_regions=[]).build().apply(model)

    def test_forward_backward_matches_without_checkpointing(self):
        pytest.importorskip("torch_remat")
        require_torch_remat()
        torch.manual_seed(42)
        baseline = _RematModel()
        remat_model = deepcopy(baseline)
        baseline_state_keys = list(baseline.state_dict())

        RematAC.Config(save_regions=["saved"]).build().apply(remat_model)

        x_BD = torch.randn(3, 4)
        expected = _run_forward_backward(baseline, x_BD)
        actual = _run_forward_backward(remat_model, x_BD)

        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        self.assertEqual(len(actual[2]), len(expected[2]))
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

        remat_block = remat_model.layers["0"]
        assert isinstance(remat_block, _RematBlock)
        self.assertEqual(remat_block.saved.num_forwards, 1)
        self.assertEqual(remat_block.recomputed.num_forwards, 2)
        self.assertTrue(remat_block.__dict__["_torchtitan_recomputes_forward"])
        self.assertEqual(list(remat_model.state_dict()), baseline_state_keys)

    def test_deepseek_default_regions_match_model(self):
        pytest.importorskip("torch_remat")
        require_torch_remat()
        from torchtitan.models.deepseek_v3 import model_registry

        model_spec = model_registry("debugmodel")
        with torch.device("meta"):
            model = model_spec.model.build()
        baseline_state_keys = list(model.state_dict())

        RematAC.Config().build().apply(model)

        self.assertEqual(list(model.state_dict()), baseline_state_keys)


if __name__ == "__main__":
    unittest.main()
