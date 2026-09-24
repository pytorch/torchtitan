# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, cast

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_wgrad_accumulation_pass,
    insert_graph_gradient_accumulation,
)


def _mm_graph(
    dtype: torch.dtype = torch.bfloat16,
    *,
    annotate_wgrad: bool = True,
) -> fx.GraphModule:
    graph = fx.Graph()
    lhs = graph.placeholder("lhs")
    lhs.meta["val"] = torch.empty(3, 4, dtype=dtype)
    rhs = graph.placeholder("rhs")
    rhs.meta["val"] = torch.empty(4, 2, dtype=dtype)
    wgrad = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
    wgrad.meta["val"] = torch.empty(3, 2, dtype=dtype)
    if annotate_wgrad:
        wgrad.meta["custom"] = {PARAMETER_GRADIENT_FQNS_META: ("weight",)}
    graph.output((wgrad,))
    return fx.GraphModule(torch.nn.Module(), graph)


def _mxfp8_scaled_mm_v2_graph(
    *,
    keyword_arguments: bool = False,
    annotate_wgrad: bool = True,
    bias: bool = False,
    recipe: F.ScalingType = F.ScalingType.BlockWise1x32,
    out_dtype: torch.dtype = torch.bfloat16,
    contraction_dim: tuple[int, ...] = (),
    use_fast_accum: bool = False,
    producer_fanout: bool = False,
) -> tuple[
    fx.GraphModule,
    FakeTensorMode,
    tuple[torch.Tensor, ...],
    torch.Tensor,
    fx.Node,
]:
    mode = FakeTensorMode()
    with mode:
        values = {
            "lhs": torch.empty(
                128,
                128,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            ),
            "rhs": torch.empty(
                128,
                128,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            ).t(),
            "lhs_scale": torch.empty(
                512,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            ),
            "rhs_scale": torch.empty(
                512,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            ),
        }
        if bias:
            values["bias"] = torch.empty(
                128,
                device="cuda",
                dtype=torch.bfloat16,
            )

        graph = fx.Graph()
        nodes: dict[str, fx.Node] = {}
        for name, value in values.items():
            node = graph.placeholder(name)
            node.meta["val"] = value
            nodes[name] = node

        scaled_mm_arguments = (
            [nodes["lhs_scale"]],
            [recipe.value],
            [F.SwizzleType.SWIZZLE_32_4_4.value],
            [nodes["rhs_scale"]],
            [recipe.value],
            [F.SwizzleType.SWIZZLE_32_4_4.value],
            nodes.get("bias"),
            out_dtype,
            list(contraction_dim),
            use_fast_accum,
        )
        if keyword_arguments:
            wgrad = graph.call_function(
                torch.ops.aten._scaled_mm_v2.default,
                args=(nodes["lhs"], nodes["rhs"]),
                kwargs=dict(
                    zip(
                        (
                            "scale_a",
                            "recipe_a",
                            "swizzle_a",
                            "scale_b",
                            "recipe_b",
                            "swizzle_b",
                            "bias",
                            "out_dtype",
                            "contraction_dim",
                            "use_fast_accum",
                        ),
                        scaled_mm_arguments,
                        strict=True,
                    )
                ),
            )
        else:
            wgrad = graph.call_function(
                torch.ops.aten._scaled_mm_v2.default,
                args=(nodes["lhs"], nodes["rhs"], *scaled_mm_arguments),
            )
        wgrad.meta["val"] = torch.empty(
            128,
            128,
            device="cuda",
            dtype=out_dtype,
        )
        if annotate_wgrad:
            wgrad.meta["custom"] = {PARAMETER_GRADIENT_FQNS_META: ("weight",)}
        extra_output = (
            graph.call_function(torch.ops.aten.alias.default, args=(wgrad,))
            if producer_fanout
            else None
        )
        graph.output((wgrad,) if extra_output is None else (wgrad, extra_output))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        (accumulator,) = insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=wgrad.meta["val"].device,
        )

    assert accumulator is not None
    return gm, mode, (*values.values(), accumulator), accumulator, wgrad


class TestWgradAccumulation(unittest.TestCase):
    def test_accumulator_metadata_has_independent_tensor_identity(self) -> None:
        with FakeTensorMode():
            gm = _mm_graph()
            insert_graph_gradient_accumulation(
                gm,
                num_param_grads=1,
                device=torch.device("cpu"),
            )

        (wgrad,) = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )
        (accumulator,) = [
            node
            for node in gm.graph.find_nodes(op="placeholder")
            if node.name.startswith("graph_runtime_grad_accumulator_")
        ]
        wgrad_value = wgrad.meta["val"]
        accumulator_value = accumulator.meta["val"]

        self.assertIsNot(accumulator_value, wgrad_value)
        self.assertEqual(accumulator_value.shape, wgrad_value.shape)
        self.assertEqual(accumulator_value.stride(), wgrad_value.stride())
        self.assertEqual(accumulator_value.dtype, wgrad_value.dtype)
        self.assertEqual(accumulator_value.device, wgrad_value.device)
        self.assertIs(accumulator_value.fake_mode, wgrad_value.fake_mode)

    def test_graph_accumulator_reuses_one_buffer(self) -> None:
        gm = _mm_graph(torch.float64)
        (accumulator,) = insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=torch.device("cpu"),
        )
        assert accumulator is not None
        accumulator.zero_()
        lhs = torch.randn(3, 4, dtype=torch.float64)
        rhs = torch.randn(4, 2, dtype=torch.float64)

        first = gm(lhs, rhs, accumulator)[0]
        second = gm(lhs, rhs, accumulator)[0]

        self.assertIs(first, accumulator)
        self.assertIs(second, accumulator)
        torch.testing.assert_close(accumulator, 2 * (lhs @ rhs))

    def test_graph_accumulator_preserves_outputs_before_parameter_grads(self) -> None:
        gm = _mm_graph(torch.float64)
        output = gm.graph.find_nodes(op="output")[0]
        (wgrad,) = output.args[0]
        lhs = gm.graph.find_nodes(op="placeholder")[0]
        with gm.graph.inserting_before(output):
            loss = gm.graph.call_function(torch.ops.aten.sum.default, args=(lhs,))
        loss.meta["val"] = torch.empty((), dtype=torch.float64)
        output.args = ((loss, wgrad),)
        gm.graph.lint()
        gm.recompile()

        (accumulator,) = insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            param_grad_output_start=1,
            device=torch.device("cpu"),
        )
        assert accumulator is not None
        accumulator.zero_()
        lhs_value = torch.randn(3, 4, dtype=torch.float64)
        rhs_value = torch.randn(4, 2, dtype=torch.float64)

        actual_loss, actual_grad = gm(lhs_value, rhs_value, accumulator)

        torch.testing.assert_close(actual_loss, lhs_value.sum())
        self.assertIs(actual_grad, accumulator)
        torch.testing.assert_close(actual_grad, lhs_value @ rhs_value)

    def test_duplicate_gradient_outputs_share_one_accumulator(self) -> None:
        gm = _mm_graph(torch.float64)
        output = gm.graph.find_nodes(op="output")[0]
        (wgrad,) = output.args[0]
        output.args = ((wgrad, wgrad),)
        gm.recompile()

        first, second = insert_graph_gradient_accumulation(
            gm,
            num_param_grads=2,
            device=torch.device("cpu"),
        )

        self.assertIs(first, second)
        self.assertEqual(len(gm.graph.find_nodes(op="placeholder")), 3)

    def test_bf16_mm_accumulation_fuses_to_addmm(self) -> None:
        gm = _mm_graph()
        (accumulator,) = insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=torch.device("cpu"),
        )

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten.addmm_.default, targets)
        self.assertNotIn(torch.ops.aten.mm.default, targets)
        self.assertNotIn(torch.ops.aten.add_.Tensor, targets)

        assert accumulator is not None
        accumulator.zero_()
        lhs = torch.randn(3, 4, dtype=torch.bfloat16)
        rhs = torch.randn(4, 2, dtype=torch.bfloat16)
        expected = torch.addmm(torch.zeros_like(accumulator), lhs, rhs)
        (actual,) = gm(lhs, rhs, accumulator)
        self.assertIs(actual, accumulator)
        torch.testing.assert_close(actual, expected)

    def test_mxfp8_scaled_mm_v2_positional_accumulation_fuses(self) -> None:
        gm, mode, inputs, accumulator, wgrad = _mxfp8_scaled_mm_v2_graph()

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten._scaled_addmm_.default, targets)
        self.assertNotIn(torch.ops.aten._scaled_mm_v2.default, targets)
        self.assertNotIn(torch.ops.aten.add_.Tensor, targets)
        self.assertEqual(wgrad.target, torch.ops.aten._scaled_addmm_.default)

        placeholders = {
            node.name: node for node in gm.graph.find_nodes(op="placeholder")
        }
        arguments = cast(Any, wgrad.args)
        self.assertIs(arguments[0], placeholders["graph_runtime_grad_accumulator_0"])
        self.assertIs(arguments[1], placeholders["lhs"])
        self.assertIs(arguments[2], placeholders["rhs"])
        self.assertIs(arguments[3][0], placeholders["lhs_scale"])
        self.assertEqual(
            tuple(arguments[4]),
            (F.ScalingType.BlockWise1x32.value,),
        )
        self.assertEqual(
            tuple(arguments[5]),
            (F.SwizzleType.SWIZZLE_32_4_4.value,),
        )
        self.assertIs(arguments[6][0], placeholders["rhs_scale"])
        self.assertEqual(
            tuple(arguments[7]),
            (F.ScalingType.BlockWise1x32.value,),
        )
        self.assertEqual(
            tuple(arguments[8]),
            (F.SwizzleType.SWIZZLE_32_4_4.value,),
        )
        self.assertEqual(arguments[9], [])
        self.assertEqual(
            wgrad.kwargs,
            {"beta": 1, "alpha": 1, "use_fast_accum": False},
        )
        self.assertEqual(
            wgrad.meta["original_aten"],
            torch.ops.aten._scaled_addmm_.default,
        )

        with mode:
            (actual,) = gm(*inputs)
        self.assertIs(actual, accumulator)

    def test_mxfp8_scaled_mm_v2_keyword_arguments_fuse(self) -> None:
        gm, mode, inputs, accumulator, wgrad = _mxfp8_scaled_mm_v2_graph(
            keyword_arguments=True,
            contraction_dim=(-1, -2),
            use_fast_accum=True,
        )

        fuse_wgrad_accumulation_pass(gm)

        self.assertEqual(wgrad.target, torch.ops.aten._scaled_addmm_.default)
        self.assertEqual(wgrad.args[9], [-1, -2])
        self.assertEqual(
            wgrad.kwargs,
            {"beta": 1, "alpha": 1, "use_fast_accum": True},
        )
        with mode:
            (actual,) = gm(*inputs)
        self.assertIs(actual, accumulator)

    def test_ineligible_scaled_mm_v2_keeps_explicit_accumulation(self) -> None:
        cases: tuple[dict[str, Any], ...] = (
            {"bias": True},
            {"recipe": F.ScalingType.TensorWise},
            {"out_dtype": torch.float16},
            {"annotate_wgrad": False},
            {"producer_fanout": True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                gm, _mode, _inputs, _accumulator, _wgrad = _mxfp8_scaled_mm_v2_graph(
                    **overrides
                )

                fuse_wgrad_accumulation_pass(gm)

                targets = {node.target for node in gm.graph.nodes}
                self.assertIn(torch.ops.aten._scaled_mm_v2.default, targets)
                self.assertIn(torch.ops.aten.add_.Tensor, targets)
                self.assertNotIn(torch.ops.aten._scaled_addmm_.default, targets)

    def test_non_unit_accumulation_alpha_keeps_scaled_mm_v2(self) -> None:
        gm, _mode, _inputs, _accumulator, _wgrad = _mxfp8_scaled_mm_v2_graph()
        (sink,) = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.add_.Tensor,
        )
        sink.kwargs = {"alpha": 2}
        gm.recompile()

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten._scaled_mm_v2.default, targets)
        self.assertIn(torch.ops.aten.add_.Tensor, targets)
        self.assertNotIn(torch.ops.aten._scaled_addmm_.default, targets)

    def test_legacy_mxfp8_scaled_mm_accumulation_fuses(self) -> None:
        gm, mode, inputs, accumulator, wgrad = _mxfp8_scaled_mm_v2_graph()
        arguments = cast(Any, wgrad.args)
        wgrad.target = torch.ops.aten._scaled_mm.default
        wgrad.args = (
            arguments[0],
            arguments[1],
            arguments[2][0],
            arguments[5][0],
        )
        wgrad.kwargs = {"out_dtype": torch.bfloat16}
        gm.recompile()

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten._scaled_addmm_.default, targets)
        self.assertNotIn(torch.ops.aten._scaled_mm.default, targets)
        self.assertNotIn(torch.ops.aten.add_.Tensor, targets)
        rewritten_arguments = cast(Any, wgrad.args)
        placeholders = {
            node.name: node for node in gm.graph.find_nodes(op="placeholder")
        }
        self.assertIs(
            rewritten_arguments[0],
            placeholders["graph_runtime_grad_accumulator_0"],
        )
        self.assertEqual(rewritten_arguments[3], [placeholders["lhs_scale"]])
        self.assertEqual(
            tuple(rewritten_arguments[4]),
            (F.ScalingType.BlockWise1x32.value,),
        )
        self.assertEqual(
            tuple(rewritten_arguments[5]),
            (F.SwizzleType.SWIZZLE_32_4_4.value,),
        )
        self.assertEqual(rewritten_arguments[6], [placeholders["rhs_scale"]])
        self.assertEqual(
            tuple(rewritten_arguments[7]),
            (F.ScalingType.BlockWise1x32.value,),
        )
        self.assertEqual(
            tuple(rewritten_arguments[8]),
            (F.SwizzleType.SWIZZLE_32_4_4.value,),
        )
        self.assertEqual(rewritten_arguments[9], [])
        self.assertEqual(
            wgrad.kwargs,
            {"beta": 1, "alpha": 1, "use_fast_accum": False},
        )
        with mode:
            (actual,) = gm(*inputs)
        self.assertIs(actual, accumulator)

    def test_legacy_non_mxfp8_scaled_mm_keeps_explicit_accumulation(self) -> None:
        gm, mode, _inputs, _accumulator, wgrad = _mxfp8_scaled_mm_v2_graph()
        arguments = cast(Any, wgrad.args)
        wgrad.target = torch.ops.aten._scaled_mm.default
        wgrad.args = (
            arguments[0],
            arguments[1],
            arguments[2][0],
            arguments[5][0],
        )
        wgrad.kwargs = {"out_dtype": torch.bfloat16}
        with mode:
            arguments[2][0].meta["val"] = torch.ones((), device="cuda")
        gm.recompile()

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten._scaled_mm.default, targets)
        self.assertIn(torch.ops.aten.add_.Tensor, targets)
        self.assertNotIn(torch.ops.aten._scaled_addmm_.default, targets)

    def test_non_bf16_mm_keeps_explicit_accumulation(self) -> None:
        gm = _mm_graph(torch.float32)
        insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=torch.device("cpu"),
        )

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten.mm.default, targets)
        self.assertIn(torch.ops.aten.add_.Tensor, targets)
        self.assertNotIn(torch.ops.aten.addmm_.default, targets)

    def test_unannotated_mm_keeps_explicit_accumulation(self) -> None:
        gm = _mm_graph(annotate_wgrad=False)
        insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=torch.device("cpu"),
        )

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten.mm.default, targets)
        self.assertIn(torch.ops.aten.add_.Tensor, targets)
        self.assertNotIn(torch.ops.aten.addmm_.default, targets)


if __name__ == "__main__":
    unittest.main()
