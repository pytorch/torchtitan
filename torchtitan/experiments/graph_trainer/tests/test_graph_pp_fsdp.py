# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.graph_pp import (
    GraphPPInputSource,
    split_backward_fsdp_collectives,
    split_forward_fsdp_collectives,
)


def _targets(gm: fx.GraphModule) -> set[object]:
    return {node.target for node in gm.graph.nodes if node.op == "call_function"}


def _placeholder_names(gm: fx.GraphModule) -> list[str]:
    return [node.name for node in gm.graph.find_nodes(op="placeholder")]


def _flat_input_sources(*names: str) -> tuple[GraphPPInputSource, ...]:
    return tuple(
        GraphPPInputSource(name=name, kind="flat_input", index=index)
        for index, name in enumerate(names)
    )


def _make_forward_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(param, 1, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(wait, x))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_forward_fsdp_view_graph() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(param, 1, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    view = graph.call_function(torch.ops.aten.view.default, args=(wait, [4]))
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(view, x))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_forward_fsdp_split_cat_graph() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(param, 1, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(wait, 2, 0))
    left = graph.call_function(operator.getitem, args=(split, 0))
    right = graph.call_function(operator.getitem, args=(split, 1))
    cat = graph.call_function(torch.ops.aten.cat.default, args=([left, right], 0))
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(cat, x))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_forward_malformed_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(param, 1, "0"),
    )
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(all_gather, x))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_forward_no_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(param, x))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_backward_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(grad, "sum", 1, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    graph.output((wait,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_backward_fsdp_graph_with_cast() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    cast = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(grad,),
        kwargs={"dtype": torch.float32},
    )
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(cast, "sum", 1, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    graph.output((wait,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_backward_no_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    out = graph.call_function(torch.ops.aten.neg.default, args=(grad,))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_backward_fsdp_graph_with_repeated_metadata() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    metadata = graph.placeholder("metadata")
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(grad, "sum", 1, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    graph.output((wait, metadata, wait, metadata))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


class GraphPPFSDPCollectiveSplitTest(unittest.TestCase):
    def test_forward_split_extracts_unshard_graph(self) -> None:
        split = split_forward_fsdp_collectives(
            _make_forward_fsdp_graph(),
            num_params=1,
            fwd_input_sources=_flat_input_sources("param", "x"),
        )

        self.assertIsNotNone(split.unshard_module)
        assert split.unshard_module is not None
        self.assertEqual(_placeholder_names(split.unshard_module), ["param"])
        self.assertEqual(len(_placeholder_names(split.fw_no_fsdp_module)), 2)
        self.assertEqual(_placeholder_names(split.fw_no_fsdp_module)[1], "x")
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _targets(split.unshard_module),
        )
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _targets(split.fw_no_fsdp_module),
        )
        self.assertEqual(len(split.unshard_output_names), 1)
        self.assertEqual(len(split.fw_no_fsdp_input_sources), 2)
        self.assertEqual(len(split.fw_no_fsdp_output_names), 1)

    def test_forward_split_leaves_views_in_no_fsdp_graph(self) -> None:
        split = split_forward_fsdp_collectives(
            _make_forward_fsdp_view_graph(),
            num_params=1,
            fwd_input_sources=_flat_input_sources("param", "x"),
        )

        self.assertIsNotNone(split.unshard_module)
        assert split.unshard_module is not None
        self.assertNotIn(torch.ops.aten.view.default, _targets(split.unshard_module))
        self.assertIn(torch.ops.aten.view.default, _targets(split.fw_no_fsdp_module))

    def test_forward_split_keeps_split_cat_reconstruction_in_unshard(self) -> None:
        split = split_forward_fsdp_collectives(
            _make_forward_fsdp_split_cat_graph(),
            num_params=1,
            fwd_input_sources=_flat_input_sources("param", "x"),
        )

        self.assertIsNotNone(split.unshard_module)
        assert split.unshard_module is not None
        self.assertIn(torch.ops.aten.split.Tensor, _targets(split.unshard_module))
        self.assertIn(torch.ops.aten.cat.default, _targets(split.unshard_module))
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _targets(split.fw_no_fsdp_module),
        )

    def test_forward_split_requires_wait_after_all_gather(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected wait_tensor"):
            split_forward_fsdp_collectives(
                _make_forward_malformed_fsdp_graph(),
                num_params=1,
                fwd_input_sources=_flat_input_sources("param", "x"),
            )

    def test_forward_split_no_fsdp_is_noop(self) -> None:
        gm = _make_forward_no_fsdp_graph()
        split = split_forward_fsdp_collectives(
            gm,
            num_params=1,
            fwd_input_sources=_flat_input_sources("param", "x"),
        )

        self.assertIsNone(split.unshard_module)
        self.assertIs(split.fw_no_fsdp_module, gm)
        self.assertEqual(_placeholder_names(split.fw_no_fsdp_module), ["param", "x"])

    def test_backward_split_extracts_reduce_grad_graph(self) -> None:
        split = split_backward_fsdp_collectives(
            _make_backward_fsdp_graph(),
            num_param_grads=1,
        )

        self.assertIsNotNone(split.reduce_grad_module)
        assert split.reduce_grad_module is not None
        self.assertEqual(_placeholder_names(split.bw_no_fsdp_module), ["grad"])
        self.assertEqual(_placeholder_names(split.reduce_grad_module), ["grad"])
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _targets(split.bw_no_fsdp_module),
        )
        self.assertIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _targets(split.reduce_grad_module),
        )
        self.assertEqual(len(split.bw_no_fsdp_output_names), 1)
        self.assertEqual(len(split.reduce_grad_input_names), 1)

    def test_backward_split_keeps_pre_reduce_cast_in_no_fsdp_graph(self) -> None:
        split = split_backward_fsdp_collectives(
            _make_backward_fsdp_graph_with_cast(),
            num_param_grads=1,
        )

        self.assertIsNotNone(split.reduce_grad_module)
        assert split.reduce_grad_module is not None
        self.assertIn(
            torch.ops.aten._to_copy.default,
            _targets(split.bw_no_fsdp_module),
        )
        self.assertNotIn(
            torch.ops.aten._to_copy.default,
            _targets(split.reduce_grad_module),
        )
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _targets(split.bw_no_fsdp_module),
        )
        self.assertIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _targets(split.reduce_grad_module),
        )

    def test_backward_split_deduplicates_repeated_reduce_inputs(self) -> None:
        split = split_backward_fsdp_collectives(
            _make_backward_fsdp_graph_with_repeated_metadata(),
            num_param_grads=4,
        )

        self.assertIsNotNone(split.reduce_grad_module)
        assert split.reduce_grad_module is not None
        self.assertEqual(
            _placeholder_names(split.reduce_grad_module), ["grad", "metadata"]
        )
        self.assertEqual(
            split.bw_no_fsdp_output_names,
            ("grad", "metadata", "grad", "metadata"),
        )
        self.assertEqual(split.reduce_grad_input_names, ("grad", "metadata"))

    def test_backward_split_no_fsdp_is_noop(self) -> None:
        gm = _make_backward_no_fsdp_graph()
        split = split_backward_fsdp_collectives(gm, num_param_grads=1)

        self.assertIsNone(split.reduce_grad_module)
        self.assertIs(split.bw_no_fsdp_module, gm)
        self.assertEqual(_placeholder_names(split.bw_no_fsdp_module), ["grad"])


if __name__ == "__main__":
    unittest.main()
