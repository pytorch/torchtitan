# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial
from unittest.mock import patch

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtitan.models.common.linear as linear_module
import torchtitan.models.common.vision_encoder as vision_encoder_module
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.vision_encoder import InvariantRowParallelLinear
from torchtitan.protocols.module import Module


class TestLinear(unittest.TestCase):
    """Tests for the Linear class used in the codebase."""

    def test_config_build(self):
        """Linear.Config.build() creates a working linear."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertIsInstance(linear, nn.Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))
        self.assertIsNone(linear.bias)

    def test_config_build_with_bias(self):
        """Linear.Config(bias=True).build() creates a linear with bias."""
        config = Linear.Config(in_features=32, out_features=16, bias=True)
        linear = config.build()
        self.assertIsNotNone(linear.bias)
        self.assertEqual(linear.bias.shape, torch.Size([16]))

    def test_config_build_without_fields_raises(self):
        """Linear.Config() raises TypeError when required features are not provided."""
        with self.assertRaises(TypeError):
            Linear.Config()

    def test_init_states(self):
        """init_states re-initializes the weight tensor."""
        config = Linear.Config(
            in_features=16,
            out_features=8,
            param_init={
                "weight": partial(nn.init.trunc_normal_, std=0.02),
                "bias": nn.init.zeros_,
            },
        )
        linear = config.build()

        with torch.no_grad():
            nn.init.zeros_(linear.weight)
            self.assertTrue(torch.all(linear.weight == 0))
            linear.init_states()
            self.assertFalse(torch.all(linear.weight == 0))

    def test_custom_init_std(self):
        """Linear respects custom mean and std."""
        config = Linear.Config(
            in_features=1000,
            out_features=500,
            param_init={
                "weight": partial(nn.init.normal_, mean=0.1, std=0.02),
                "bias": nn.init.zeros_,
            },
        )
        linear = config.build()

        torch.manual_seed(42)
        with torch.no_grad():
            linear.init_states()
        # With large amount of samples (1000 * 500) the sample statistics should
        # be close to the requested values. places=3 checks within 0.0005, which
        # is well within statistical tolerance for this sample size.
        self.assertAlmostEqual(linear.weight.mean().item(), 0.1, places=3)
        self.assertAlmostEqual(linear.weight.std().item(), 0.02, places=3)

    def test_forward(self):
        """Forward pass works through nn.Linear's implementation."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        x = torch.randn(2, 10, 32)
        out = linear(x)
        self.assertEqual(out.shape, torch.Size([2, 10, 16]))

    def test_shared_config_builds_independent_instances(self):
        """A single Linear.Config can build multiple independent linears."""
        cfg1 = Linear.Config(in_features=32, out_features=16)
        l1 = cfg1.build()
        cfg2 = Linear.Config(in_features=64, out_features=8)
        l2 = cfg2.build()
        self.assertIsNot(l1, l2)
        self.assertEqual(l1.weight.shape, torch.Size([16, 32]))
        self.assertEqual(l2.weight.shape, torch.Size([8, 64]))

    def test_isinstance_checks(self):
        """Linear is instance of nn.Linear, and Module."""
        config = Linear.Config(in_features=8, out_features=4)
        linear = config.build()
        self.assertIsInstance(linear, nn.Linear)
        self.assertIsInstance(linear, Module)

    def test_default_bias_false(self):
        """Linear.Config defaults to bias=False."""
        config = Linear.Config(in_features=4, out_features=4)
        self.assertFalse(config.bias)

    def test_direct_construction(self):
        """Linear can be constructed directly (Flux-style, non-Configurable parents)."""
        config = Linear.Config(in_features=32, out_features=16, bias=True)
        linear = Linear(config)
        self.assertIsInstance(linear, Linear)
        self.assertIsNotNone(linear.bias)

    def test_config_pre_specified_build(self):
        """Linear.Config with both fields pre-specified builds with no kwargs."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))

    def test_config_partial_pre_specified(self):
        """Linear.Config with fields specified at construction builds correctly."""
        config = Linear.Config(in_features=32, out_features=16)
        linear = config.build()
        self.assertIsInstance(linear, Linear)
        self.assertEqual(linear.weight.shape, torch.Size([16, 32]))


class TestGroupedLinear(unittest.TestCase):
    def test_num_linears_preserves_projection_axis_without_copy(self):
        """Multiple projections retain a zero-copy logical output axis."""
        grouped = GroupedLinear.Config(
            group_size=3,
            in_features=4,
            out_features=8,
            num_linears=2,
        ).build()

        self.assertEqual(grouped.weight.shape, torch.Size([3, 2, 8, 4]))
        weight_EOI = grouped.weight.flatten(1, -2)
        self.assertEqual(weight_EOI.shape, torch.Size([3, 16, 4]))
        self.assertEqual(
            weight_EOI.untyped_storage().data_ptr(),
            grouped.weight.untyped_storage().data_ptr(),
        )

    def test_forward_restores_projection_axis(self):
        """Forward restores the projection axis after the grouped GEMM seam."""

        class StubGroupedLinear(GroupedLinear):
            def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
                del offsets_E
                return input_RI.new_zeros(input_RI.shape[0], weight_EOI.shape[1])

        grouped = StubGroupedLinear(
            GroupedLinear.Config(
                group_size=2,
                in_features=4,
                out_features=8,
                num_linears=2,
            )
        )
        output_R2F = grouped(
            torch.randn(5, 4),
            torch.tensor([2, 5], dtype=torch.int32),
        )

        self.assertEqual(output_R2F.shape, torch.Size([5, 2, 8]))


class TestTensorParallelLinearSpmdTypes(unittest.TestCase):
    def test_collective_types_follow_dense_sp_state(self):
        input = torch.randn(3, 4)
        tp_group = object()

        for dense_sp_enabled, expected_type in (
            (False, spmd.I),
            (True, spmd.S(0)),
        ):
            with self.subTest(dense_sp_enabled=dense_sp_enabled):
                column = ColumnParallelLinear.Config(
                    in_features=4,
                    out_features=2,
                ).build()
                row = RowParallelLinear.Config(
                    in_features=4,
                    out_features=2,
                    bias=True,
                ).build()

                with (
                    patch.object(
                        linear_module,
                        "spmd_mesh_group",
                        return_value=tp_group,
                    ),
                    patch.object(
                        linear_module,
                        "spmd_dense_sp_enabled",
                        return_value=dense_sp_enabled,
                    ),
                    patch.object(
                        linear_module.spmd,
                        "convert",
                        side_effect=lambda tensor, *_args, **_kwargs: tensor,
                    ) as convert,
                    patch.object(
                        linear_module.spmd,
                        "redistribute",
                        side_effect=lambda tensor, *_args, **_kwargs: tensor,
                    ) as redistribute,
                ):
                    column(input)
                    assert redistribute.call_args.kwargs["src"] == expected_type
                    assert redistribute.call_args.kwargs["dst"] == spmd.R

                    row(input)
                    assert convert.call_args.kwargs["src"] == spmd.I
                    assert convert.call_args.kwargs["dst"] == spmd.P
                    assert redistribute.call_args.kwargs["src"] == spmd.P
                    assert redistribute.call_args.kwargs["dst"] == expected_type

    def test_vision_row_parallel_output_is_always_invariant(self):
        input = torch.randn(3, 4)
        tp_group = object()
        linear = InvariantRowParallelLinear.Config(
            in_features=4,
            out_features=2,
            bias=True,
        ).build()

        with (
            patch.object(
                vision_encoder_module,
                "spmd_mesh_group",
                return_value=tp_group,
            ),
            patch.object(
                linear_module.spmd,
                "convert",
                side_effect=lambda tensor, *_args, **_kwargs: tensor,
            ) as convert,
            patch.object(
                vision_encoder_module.spmd,
                "redistribute",
                side_effect=lambda tensor, *_args, **_kwargs: tensor,
            ) as redistribute,
        ):
            linear(input)

        assert convert.call_args.kwargs["src"] == spmd.I
        assert convert.call_args.kwargs["dst"] == spmd.P
        assert redistribute.call_args.kwargs["src"] == spmd.P
        assert redistribute.call_args.kwargs["dst"] == spmd.I


class TestBiasedRowParallelLinearDistributed(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_tp_forward_and_bias_gradient_match_unsharded(self):
        mesh = init_device_mesh(self.device_type, (2,), mesh_dim_names=("tp",))
        tp_group = mesh.get_group("tp")
        torch.manual_seed(42)

        input = torch.randn(3, 4, device=self.device_type)
        weight = torch.randn(2, 4, device=self.device_type)
        bias = torch.randn(2, device=self.device_type)

        expected_input = input.detach().clone().requires_grad_()
        expected_weight = weight.detach().clone().requires_grad_()
        expected_bias = bias.detach().clone().requires_grad_()
        expected = F.linear(expected_input, expected_weight, expected_bias)
        expected.sum().backward()

        input_dtensor = distribute_tensor(input, mesh, (Shard(1),))
        weight_dtensor = distribute_tensor(weight, mesh, (Shard(1),))
        local_input = input_dtensor.to_local().detach().requires_grad_()
        for linear_cls in (RowParallelLinear, InvariantRowParallelLinear):
            with self.subTest(linear_cls=linear_cls.__name__):
                linear = (
                    linear_cls.Config(
                        in_features=4,
                        out_features=2,
                        bias=True,
                    )
                    .build()
                    .to(self.device_type)
                )
                linear.weight = nn.Parameter(weight_dtensor.to_local())
                linear.bias = nn.Parameter(bias.detach().clone())
                with set_current_spmd_mesh(mesh), typecheck(local=False):
                    linear._parameters["weight"] = spmd.assert_type(
                        linear.weight, {tp_group: spmd.S(1)}
                    )
                    linear._parameters["bias"] = spmd.assert_type(
                        linear.bias, {tp_group: spmd.I}
                    )
                    typed_input = spmd.assert_type(local_input, {tp_group: spmd.S(1)})
                    actual = linear(typed_input)
                    actual.sum().backward()

                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(linear.bias.grad, expected_bias.grad)


if __name__ == "__main__":
    unittest.main()
