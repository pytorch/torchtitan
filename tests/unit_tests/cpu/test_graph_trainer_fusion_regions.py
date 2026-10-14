# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import operator
import unittest
from types import SimpleNamespace

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

from torchtitan.config import TrainingConfig
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.fusion_regions import (
    find_fusion_region,
    FUSION_REGIONS_META,
)
from torchtitan.experiments.graph_trainer.passes import (
    _get_pass_name,
    apply_graph_passes,
    compile_time_passes,
    find_fusion_region as pipeline_find_fusion_region,
)


@torch.library.custom_op("graph_trainer_fusion_test::pure", mutates_args=())
def _pure(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("graph_trainer_fusion_test::ordered", mutates_args=())
def _ordered(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


_ordered.register_effect(torch.library.EffectType.ORDERED)


def _opaque(x):
    return x


def _module(graph, root=None):
    return torch.fx.GraphModule(torch.nn.Module() if root is None else root, graph)


def _names(nodes):
    return tuple(node.name for node in nodes)


def _bytes(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).clone()


class _NoTensorExecution(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"Region analysis executed tensor operation {func}")


class TestGraphTrainerFusionRegions(unittest.TestCase):
    def test_residual_regions_preserve_saved_values_and_external_io_order(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        weight = graph.placeholder("weight")
        saved = graph.call_function(torch.ops.aten.sin.default, (x,))
        activation = graph.call_function(torch.ops.aten.cos.default, (saved,))
        gemm = graph.call_function(torch.ops.aten.mm.default, (activation, weight))
        hidden = graph.call_function(torch.ops.aten.relu.default, (gemm,))
        output = graph.call_function(torch.ops.aten.add.Tensor, (hidden, saved))
        graph.output((output, saved))
        gm = _module(graph)

        self.assertIs(find_fusion_region(gm), gm)
        summary = gm.meta[FUSION_REGIONS_META]
        self.assertEqual(
            summary["regions"],
            (
                {
                    "id": 0,
                    "nodes": _names((saved, activation)),
                    "inputs": (x.name,),
                    "outputs": _names((saved, activation)),
                },
                {
                    "id": 1,
                    "nodes": _names((hidden, output)),
                    "inputs": _names((saved, gemm)),
                    "outputs": (output.name,),
                },
            ),
        )
        self.assertEqual(
            summary["boundaries"], ({"node": gemm.name, "kinds": ("gemm",)},)
        )

    def test_unrelated_branches_sharing_a_placeholder_stay_separate(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        left = graph.call_function(torch.ops.aten.sin.default, (x,))
        right = graph.call_function(torch.ops.aten.cos.default, (x,))
        left_out = graph.call_function(torch.ops.aten.mul.Tensor, (left, left))
        right_out = graph.call_function(torch.ops.aten.add.Tensor, (right, right))
        graph.output((left_out, right_out))
        gm = _module(graph)
        find_fusion_region(gm)
        regions = gm.meta[FUSION_REGIONS_META]["regions"]
        self.assertEqual(
            [region["nodes"] for region in regions],
            [_names((left, left_out)), _names((right, right_out))],
        )
        self.assertEqual(
            [region["inputs"] for region in regions], [(x.name,), (x.name,)]
        )

    def test_connected_graph_without_boundaries_is_one_region(self):
        gm = make_fx(lambda x: (x.sin() + x.cos()).relu())(torch.randn(4))
        find_fusion_region(gm)
        summary = gm.meta[FUSION_REGIONS_META]
        self.assertEqual(summary["boundaries"], ())
        self.assertEqual(len(summary["regions"]), 1)
        self.assertEqual(
            summary["regions"][0]["nodes"],
            _names(node for node in gm.graph.nodes if node.op == "call_function"),
        )

    def test_identity_and_boundary_only_graphs_have_no_candidate_regions(self):
        identity = torch.fx.symbolic_trace(torch.nn.Identity())
        find_fusion_region(identity)
        self.assertEqual(
            identity.meta[FUSION_REGIONS_META], {"regions": (), "boundaries": ()}
        )
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        weight = graph.placeholder("weight")
        first = graph.call_function(torch.ops.aten.mm.default, (x, weight))
        second = graph.call_function(torch.ops.aten.mm.default, (first, weight))
        graph.output(second)
        gm = _module(graph)
        find_fusion_region(gm)
        self.assertEqual(gm.meta[FUSION_REGIONS_META]["regions"], ())
        self.assertEqual(
            [entry["node"] for entry in gm.meta[FUSION_REGIONS_META]["boundaries"]],
            list(_names((first, second))),
        )

    def test_compute_and_collective_operator_families_are_boundaries(self):
        cases = (
            (torch.ops.aten.mm.default, "gemm"),
            (torch.ops.aten.mm.dtype, "gemm"),
            (torch.ops.aten.addmm.default, "gemm"),
            (torch.ops.aten.addmm_.default, "gemm"),
            (torch.ops.aten._addmm_activation.default, "gemm"),
            (torch.ops.aten.bmm.default, "gemm"),
            (torch.ops.aten.addbmm_.default, "gemm"),
            (torch.ops.aten.baddbmm.default, "gemm"),
            (torch.ops.aten.baddbmm_.default, "gemm"),
            (torch.ops.aten.matmul_backward.default, "gemm"),
            (torch.ops.aten.linear_backward.default, "gemm"),
            (torch.ops.aten._scaled_mm.default, "gemm"),
            (torch.ops.aten._scaled_mm_v2.default, "gemm"),
            (torch.ops.aten._scaled_mm_v2.out, "gemm"),
            (torch.ops.aten._scaled_addmm.default, "gemm"),
            (torch.ops.aten._scaled_addmm_.default, "gemm"),
            (torch.ops.aten._grouped_mm.default, "gemm"),
            (torch.ops.aten._scaled_grouped_mm.default, "gemm"),
            (torch.ops.aten._scaled_grouped_mm_v2.default, "gemm"),
            (torch.ops.aten._foreach_mm.default, "gemm"),
            (torch.ops.aten.scaled_dot_product_attention.default, "attention"),
            (torch.ops.aten._scaled_dot_product_flash_attention.default, "attention"),
            (
                torch.ops.aten._scaled_dot_product_flash_attention_backward.default,
                "attention",
            ),
            (
                torch.ops.aten._scaled_dot_product_efficient_attention.default,
                "attention",
            ),
            (
                torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
                "attention",
            ),
            (torch.ops.aten._scaled_dot_product_cudnn_attention.default, "attention"),
            (
                torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default,
                "attention",
            ),
            (torch.ops.aten._flash_attention_forward.default, "attention"),
            (torch.ops.aten._flash_attention_backward.default, "attention"),
            (torch.ops.aten._cudnn_attention_forward.default, "attention"),
            (torch.ops.aten._cudnn_attention_backward.default, "attention"),
            (torch.ops.aten._native_multi_head_attention.default, "attention"),
            (torch.ops.aten._triton_multi_head_attention.default, "attention"),
            (torch.ops.aten._triton_scaled_dot_attention.default, "attention"),
            (torch.ops.higher_order.flex_attention, "attention"),
            (torch.ops.higher_order.flex_attention_backward, "attention"),
            (torch.ops._c10d_functional.all_reduce.default, "collective"),
            (torch.ops._c10d_functional.all_gather_into_tensor.default, "collective"),
            (torch.ops._c10d_functional.reduce_scatter_tensor.default, "collective"),
            (torch.ops._c10d_functional.all_to_all_single.default, "collective"),
            (torch.ops._c10d_functional.wait_tensor.default, "collective"),
            (torch.ops._c10d_functional.isend.default, "collective"),
            (torch.ops._c10d_functional.irecv.default, "collective"),
            (torch.ops.symm_mem.fused_all_gather_matmul.default, "gemm"),
            (torch.ops.symm_mem.fused_all_gather_scaled_matmul.default, "gemm"),
            (torch.ops.symm_mem.fused_matmul_reduce_scatter.default, "gemm"),
            (torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter.default, "gemm"),
        )
        for target, kind in cases:
            with self.subTest(target=target):
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                boundary = graph.call_function(target, (x, x))
                graph.output(boundary)
                gm = _module(graph)
                find_fusion_region(gm)
                summary = gm.meta[FUSION_REGIONS_META]
                self.assertEqual(summary["regions"], ())
                self.assertEqual(summary["boundaries"][0]["node"], boundary.name)
                self.assertIn(kind, summary["boundaries"][0]["kinds"])
                if str(target).startswith("symm_mem."):
                    self.assertIn("collective", summary["boundaries"][0]["kinds"])

    def test_boundary_tuple_getitems_are_glue_with_external_tuple_input(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        attention = graph.call_function(
            torch.ops.aten._scaled_dot_product_flash_attention.default, (x, x, x)
        )
        first = graph.call_function(operator.getitem, (attention, 0))
        second = graph.call_function(operator.getitem, (attention, 1))
        output = graph.call_function(torch.ops.aten.add.Tensor, (first, second))
        graph.output((output, second))
        gm = _module(graph)
        find_fusion_region(gm)
        (region,) = gm.meta[FUSION_REGIONS_META]["regions"]
        self.assertEqual(region["nodes"], _names((first, second, output)))
        self.assertEqual(region["inputs"], (attention.name,))
        self.assertEqual(region["outputs"], _names((second, output)))

    def test_alias_reads_cannot_bridge_an_inplace_write(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        replacement = graph.placeholder("replacement")
        alias = graph.call_function(torch.ops.aten.alias.default, (x,))
        saved = graph.call_function(torch.ops.aten.clone.default, (alias,))
        write = graph.call_function(torch.ops.aten.copy_.default, (x, replacement))
        output = graph.call_function(torch.ops.aten.add.Tensor, (saved, x))
        graph.output(output)
        gm = _module(graph)
        find_fusion_region(gm)
        summary = gm.meta[FUSION_REGIONS_META]
        self.assertEqual(
            [region["nodes"] for region in summary["regions"]],
            [_names((alias, saved)), (output.name,)],
        )
        self.assertEqual(summary["boundaries"][0]["node"], write.name)
        self.assertIn("mutation", summary["boundaries"][0]["kinds"])

    def test_custom_schema_glue_and_ordered_or_opaque_fences(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        pure = graph.call_function(_pure._opoverload, (x,))
        ordered = graph.call_function(_ordered._opoverload, (pure,))
        after = graph.call_function(torch.ops.aten.sin.default, (ordered,))
        opaque = graph.call_function(_opaque, (after,))
        graph.output(opaque)
        gm = _module(graph)
        find_fusion_region(gm)
        summary = gm.meta[FUSION_REGIONS_META]
        self.assertEqual(
            [region["nodes"] for region in summary["regions"]],
            [(pure.name,), (after.name,)],
        )
        kinds = {entry["node"]: entry["kinds"] for entry in summary["boundaries"]}
        self.assertIn("effect", kinds[ordered.name])
        self.assertIn("opaque", kinds[opaque.name])

    def test_random_operator_is_a_fence_without_advancing_rng(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        before = graph.call_function(torch.ops.aten.sin.default, (x,))
        random = graph.call_function(torch.ops.aten.rand.default, ([4],))
        after = graph.call_function(torch.ops.aten.add.Tensor, (before, random))
        graph.output(after)
        gm = _module(graph)
        rng_state = torch.random.get_rng_state().clone()
        with _NoTensorExecution():
            find_fusion_region(gm)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        summary = gm.meta[FUSION_REGIONS_META]
        self.assertEqual(
            [region["nodes"] for region in summary["regions"]],
            [(before.name,), (after.name,)],
        )
        self.assertEqual(
            summary["boundaries"], ({"node": random.name, "kinds": ("effect",)},)
        )

    def test_nested_graph_modules_have_local_metadata_and_optional_recursion(self):
        root = torch.nn.Module()
        root.left = make_fx(lambda x: x.sin().cos())(torch.randn(4))
        root.right = make_fx(lambda x: x.relu() * 2)(torch.randn(4))
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        left = graph.call_module("left", (x,))
        right = graph.call_module("right", (x,))
        graph.output((left, right))
        gm = _module(graph, root)
        original_codes = {name: child.code for name, child in gm.named_modules()}
        find_fusion_region(gm, recurse=False)
        self.assertNotIn(FUSION_REGIONS_META, gm.left.meta)
        self.assertNotIn(FUSION_REGIONS_META, gm.right.meta)
        find_fusion_region(gm)
        self.assertEqual(gm.meta[FUSION_REGIONS_META]["regions"], ())
        for name, child in gm.named_modules():
            self.assertEqual(child.code, original_codes[name])
            if child is not gm:
                (region,) = child.meta[FUSION_REGIONS_META]["regions"]
                self.assertEqual(region["id"], 0)
                self.assertTrue(
                    all(
                        node_name in {node.name for node in child.graph.nodes}
                        for node_name in region["nodes"]
                    )
                )

    def test_analysis_is_read_only_idempotent_and_bitwise_identical(self):
        generator = torch.Generator().manual_seed(42)

        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4, generator=generator))
                self.register_buffer("markers", torch.tensor([-0.0, float("nan")]))

            def forward(self, x):
                activation = x.tanh()
                output = (activation @ self.weight).sin() + activation
                return output, output.square().sum(), self.markers

        model = Residual()
        x = torch.randn(3, 4, generator=generator)
        gm = make_fx(model)(x)
        self.assertEqual(len(tuple(gm.parameters())), 1)
        gm.meta["existing"] = {"preserve": True}
        original_code = gm.code
        original_nodes = tuple(gm.graph.nodes)
        original_state = [
            (node.op, node.target, node.args, dict(node.kwargs), dict(node.meta))
            for node in original_nodes
        ]
        tensors = (x, *gm.parameters(), *gm.buffers())
        original_tensors = [(tensor.data_ptr(), _bytes(tensor)) for tensor in tensors]

        def run():
            input = x.clone().requires_grad_()
            outputs = gm(input)
            gradients = torch.autograd.grad(outputs[1], (input, *gm.parameters()))
            return tuple(_bytes(tensor) for tensor in (*outputs, *gradients))

        before = run()
        with _NoTensorExecution():
            self.assertIs(find_fusion_region(gm, (x,)), gm)
        serialized = json.dumps(gm.meta[FUSION_REGIONS_META])
        with _NoTensorExecution():
            find_fusion_region(gm, (x,))
        self.assertEqual(json.dumps(gm.meta[FUSION_REGIONS_META]), serialized)
        self.assertEqual(gm.meta["existing"], {"preserve": True})
        self.assertEqual(gm.code, original_code)
        self.assertEqual(tuple(gm.graph.nodes), original_nodes)
        self.assertEqual(
            [
                (node.op, node.target, node.args, dict(node.kwargs), dict(node.meta))
                for node in gm.graph.nodes
            ],
            original_state,
        )
        for tensor, (pointer, bits) in zip(tensors, original_tensors, strict=True):
            self.assertEqual(tensor.data_ptr(), pointer)
            self.assertTrue(torch.equal(_bytes(tensor), bits))
        for actual, expected in zip(run(), before, strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_rerun_replaces_stale_metadata_after_graph_edits(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        weight = graph.placeholder("weight")
        activation = graph.call_function(torch.ops.aten.sin.default, (x,))
        changed = graph.call_function(torch.ops.aten.cos.default, (activation,))
        graph.output(changed)
        gm = _module(graph)
        find_fusion_region(gm)
        previous = gm.meta[FUSION_REGIONS_META]
        changed.target = torch.ops.aten.mm.default
        changed.args = (activation, weight)
        gm.recompile()
        find_fusion_region(gm)
        current = gm.meta[FUSION_REGIONS_META]
        self.assertIsNot(current, previous)
        self.assertEqual(previous["regions"][0]["nodes"], _names((activation, changed)))
        self.assertEqual(current["regions"][0]["nodes"], (activation.name,))
        self.assertEqual(
            current["boundaries"], ({"node": changed.name, "kinds": ("gemm",)},)
        )

    def test_pipeline_reexport_exact_disable_name_and_opt_in_order(self):
        self.assertIs(pipeline_find_fusion_region, find_fusion_region)
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        config = GraphTrainerCompileConfig(disable_passes=["find_fusion_region"])
        self.assertIs(
            apply_graph_passes(gm, (), [find_fusion_region], compile_config=config), gm
        )
        self.assertNotIn(FUSION_REGIONS_META, gm.meta)
        apply_graph_passes(
            gm,
            (),
            [find_fusion_region],
            compile_config=config,
            respect_disable_passes=False,
        )
        self.assertIn(FUSION_REGIONS_META, gm.meta)

        config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(inductor_compilation="full"),
            model=SimpleNamespace(layers=[object()]),
            parallelism=SimpleNamespace(),
            training=TrainingConfig(disable_cuda_graphs=True),
            loss=None,
        )
        traced_result = SimpleNamespace(state_fqns=[])
        default_names = [
            _get_pass_name(fn) for fn in compile_time_passes(traced_result, config)
        ]
        self.assertNotIn("find_fusion_region", default_names)
        config.compile.passes = ["find_fusion_region"]
        names = [
            _get_pass_name(fn) for fn in compile_time_passes(traced_result, config)
        ]
        self.assertEqual(names.count("find_fusion_region"), 1)
        self.assertLess(
            names.index("find_fusion_region"),
            names.index("full_inductor_compilation_pass"),
        )
        uncompiled_names = [
            _get_pass_name(fn)
            for fn in compile_time_passes(traced_result, config, include_inductor=False)
        ]
        self.assertIn("find_fusion_region", uncompiled_names)


if __name__ == "__main__":
    unittest.main()
