# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.graph_trainer.autoresearch.moe_fusion import (
    _aux_backward,
    _aux_forward,
    _router_backward,
    _selection,
    _token_counts,
    _weights,
    fuse_production_moe,
)
from torchtitan.models.common.hi_mid_lo_linear import _split_into_bf16_pieces


def _fake_input(mode, shape, *, dtype=torch.float32, device="cuda", **kwargs):
    return FakeTensor(
        mode,
        torch.empty(shape, device="meta", dtype=dtype, **kwargs),
        torch.device(device),
    )


def _trace_with_cuda_metadata(function, inputs, mode):
    """Trace CPU autograd, then attach CUDA metadata without executing kernels."""
    graph = make_fx(function)(*inputs)

    def cuda_metadata(value):
        tensor = torch.empty_strided(
            value.shape, value.stride(), dtype=value.dtype, device="meta"
        )
        return FakeTensor(mode, tensor, torch.device("cuda"))

    for node in graph.graph.nodes:
        if "val" in node.meta:
            node.meta["val"] = torch.utils._pytree.tree_map_only(
                torch.Tensor, cuda_metadata, node.meta["val"]
            )
    return graph


class TestProductionMoEFusion(unittest.TestCase):
    def test_joint_autograd_graph_preserves_gemm_and_external_outputs(self):
        def joint(logits, bias, grad_weights, aux_grad, weight):
            scores = logits.sigmoid()
            indices = _selection(scores, bias)
            weights, selected, denominator = _weights(scores, indices)
            loss = (weights * grad_weights).sum() + (scores * aux_grad).sum()
            grad = torch.autograd.grad(loss, logits)[0]
            pieces = _split_into_bf16_pieces(grad, 3, dim=1)
            return pieces @ weight, weights, selected, denominator, indices

        mode = FakeTensorMode()
        inputs = (
            _fake_input(mode, (4096, 256), device="cpu", requires_grad=True),
            _fake_input(mode, (256,), device="cpu"),
            _fake_input(mode, (4096, 8), device="cpu"),
            _fake_input(mode, (4096, 256), device="cpu"),
            _fake_input(mode, (768, 32), dtype=torch.bfloat16, device="cpu"),
        )
        with mode:
            graph = _trace_with_cuda_metadata(joint, inputs, mode)
            original_gemms = [
                node
                for node in graph.graph.nodes
                if node.target == torch.ops.aten.mm.default
            ]
            counts = fuse_production_moe(
                graph, regions=("selection", "weights", "gate_backward")
            )
        self.assertEqual(
            counts,
            {"selection": 1, "weights_forward": 1, "weights_gate_backward": 1},
        )
        self.assertTrue(all(node in graph.graph.nodes for node in original_gemms))
        output = next(node for node in graph.graph.nodes if node.op == "output")
        self.assertEqual(len(output.args[0]), 5)
        self.assertEqual(tuple(output.args[0][0].meta["val"].shape), (4096, 32))
        self.assertEqual(tuple(output.args[0][2].meta["val"].shape), (4096, 8))
        self.assertEqual(tuple(output.args[0][3].meta["val"].shape), (4096, 1))
        self.assertEqual(
            fuse_production_moe(
                graph, regions=("selection", "weights", "gate_backward")
            ),
            {"selection": 0, "weights_forward": 0, "weights_gate_backward": 0},
        )

    def test_unsupported_inputs_keep_original_selection(self):
        for shape, dtype, device, transpose in (
            ((2048, 256), torch.float32, "cuda", False),
            ((4096, 256), torch.bfloat16, "cuda", False),
            ((4096, 256), torch.float32, "cpu", False),
            ((256, 4096), torch.float32, "cuda", True),
        ):
            with self.subTest(
                shape=shape, dtype=dtype, device=device, transpose=transpose
            ):
                mode = FakeTensorMode()
                scores = _fake_input(mode, shape, dtype=dtype, device=device)
                bias = _fake_input(mode, (256,), dtype=dtype, device=device)
                with mode:
                    if transpose:
                        scores = scores.t()
                    graph = make_fx(_selection)(scores, bias)
                    counts = fuse_production_moe(graph, regions=("selection",))
                self.assertEqual(counts, {"selection": 0})
                self.assertEqual(
                    sum(
                        node.target == torch.ops.aten.topk.default
                        for node in graph.graph.nodes
                    ),
                    3,
                )

    def test_additional_intermediate_consumer_prevents_selection_fusion(self):
        mode = FakeTensorMode()
        inputs = (_fake_input(mode, (4096, 256)), _fake_input(mode, (256,)))
        with mode:
            graph = make_fx(_selection)(*inputs)
            intermediate = next(
                node
                for node in graph.graph.nodes
                if node.target == torch.ops.aten.add.Tensor
            )
            output = next(node for node in graph.graph.nodes if node.op == "output")
            output.args = ((output.args[0], intermediate),)
            graph.recompile()
            counts = fuse_production_moe(graph, regions=("selection",))
        self.assertEqual(counts, {"selection": 0})
        self.assertIn(intermediate, graph.graph.nodes)

    def test_unknown_region_is_rejected(self):
        graph = torch.fx.symbolic_trace(torch.nn.Identity())
        with self.assertRaisesRegex(ValueError, "Unknown MoE fusion regions"):
            fuse_production_moe(graph, regions=("unknown",))

    def test_aux_graph_preserves_count_update_and_gradient_gemm(self):
        def joint(scores, bias, count_buffer, grad, weight):
            indices = _selection(scores, bias)
            integer_counts, float_counts = _token_counts(scores, indices)
            count_buffer.add_(integer_counts)
            normalized = torch.nn.functional.normalize(scores, p=1, dim=-1)
            grad_scores = torch.autograd.grad(normalized.sum(0), scores, grad)[0]
            return float_counts, count_buffer, normalized, grad_scores @ weight

        mode = FakeTensorMode()
        inputs = (
            _fake_input(mode, (4096, 256), device="cpu", requires_grad=True),
            _fake_input(mode, (256,), device="cpu"),
            _fake_input(mode, (256,), dtype=torch.int64, device="cpu"),
            _fake_input(mode, (256,), device="cpu"),
            _fake_input(mode, (256, 32), device="cpu"),
        )
        with mode:
            graph = _trace_with_cuda_metadata(joint, inputs, mode)
            gemm = next(
                node
                for node in graph.graph.nodes
                if node.target == torch.ops.aten.mm.default
            )
            count_update = next(
                node
                for node in graph.graph.nodes
                if node.target == torch.ops.aten.add_.Tensor
            )
            counts = fuse_production_moe(graph, regions=("selection", "counts", "aux"))
        self.assertEqual(
            counts,
            {"selection": 1, "counts": 1, "aux_forward": 1, "aux_backward": 1},
        )
        self.assertIn(gemm, graph.graph.nodes)
        self.assertIn(count_update, graph.graph.nodes)
        self.assertEqual(count_update.args[0].op, "placeholder")
        self.assertEqual(count_update.args[1].meta["val"].dtype, torch.int64)
        self.assertFalse(
            any(
                node.target == torch.ops.aten.scalar_tensor.default
                for node in graph.graph.nodes
            )
        )
        output = next(node for node in graph.graph.nodes if node.op == "output")
        self.assertEqual(tuple(output.args[0][2].meta["val"].shape), (4096, 256))
        self.assertEqual(tuple(output.args[0][3].meta["val"].shape), (4096, 32))

    def test_arbitrary_indices_keep_scatter_count_and_gradient_semantics(self):
        def joint(scores, indices, grad_weights):
            counts = _token_counts(scores, indices)
            weights = _weights(scores, indices)[0]
            grad_scores = torch.autograd.grad(weights, scores, grad_weights)[0]
            return *counts, weights, grad_scores

        mode = FakeTensorMode()
        inputs = (
            _fake_input(mode, (4096, 256), device="cpu", requires_grad=True),
            _fake_input(mode, (4096, 8), dtype=torch.int64, device="cpu"),
            _fake_input(mode, (4096, 8), device="cpu"),
        )
        with mode:
            graph = _trace_with_cuda_metadata(joint, inputs, mode)
            counts = fuse_production_moe(graph, regions=("weights", "counts"))
        self.assertEqual(
            counts, {"weights_forward": 1, "weights_backward": 0, "counts": 0}
        )
        targets = {node.target for node in graph.graph.nodes}
        self.assertIn(torch.ops.aten.scatter_.value, targets)
        self.assertIn(torch.ops.aten.scatter_add.default, targets)

    def test_external_routing_map_consumer_prevents_count_fusion(self):
        def counts_from_topk(scores):
            indices = scores.topk(8, dim=-1).indices
            return _token_counts(scores, indices)

        mode = FakeTensorMode()
        with mode:
            graph = make_fx(counts_from_topk)(_fake_input(mode, (4096, 256)))
            routing_map = next(
                node
                for node in graph.graph.nodes
                if node.target == torch.ops.aten.scatter_.value
            )
            output = next(node for node in graph.graph.nodes if node.op == "output")
            output.args = ((*output.args[0], routing_map),)
            graph.recompile()
            counts = fuse_production_moe(graph, regions=("counts",))
        self.assertEqual(counts, {"counts": 0})
        self.assertIn(routing_map, graph.graph.nodes)

    def test_misaligned_aux_scores_keep_original_normalization(self):
        mode = FakeTensorMode()
        with mode:
            scores = _fake_input(mode, (4096 * 256 + 1,))[1:].view(4096, 256)
            graph = make_fx(_aux_forward)(scores)
            counts = fuse_production_moe(graph, regions=("aux",))
        self.assertEqual(counts, {"aux_forward": 0, "aux_backward": 0})
        self.assertTrue(
            any(
                node.target == torch.ops.aten.linalg_vector_norm.default
                for node in graph.graph.nodes
            )
        )

    def test_aux_backward_requires_broadcast_norm_and_constant_zero(self):
        for unsupported_input in ("dense_norm", "dynamic_zero", "negative_zero"):
            with self.subTest(unsupported_input=unsupported_input):
                mode = FakeTensorMode()
                with mode:
                    norm_shape = (
                        (4096, 256) if unsupported_input == "dense_norm" else (4096, 1)
                    )
                    norm = _fake_input(mode, norm_shape).expand(4096, 256)
                    inputs = (
                        _fake_input(mode, (256,)),
                        _fake_input(mode, (4096, 256)),
                        _fake_input(mode, (4096, 1)),
                        norm,
                        _fake_input(mode, ()),
                    )

                    def backward(grad, scores, raw_norm, expanded_norm, zero):
                        if unsupported_input != "dynamic_zero":
                            zero = torch.ops.aten.scalar_tensor.default(
                                -0.0 if unsupported_input == "negative_zero" else 0.0,
                                dtype=torch.float32,
                                device=scores.device,
                            )
                        return _aux_backward(
                            grad, scores, raw_norm, expanded_norm, zero
                        )

                    graph = make_fx(backward)(*inputs)
                    counts = fuse_production_moe(graph, regions=("aux",))
                self.assertEqual(counts, {"aux_forward": 0, "aux_backward": 0})
                self.assertTrue(
                    any(
                        node.target == torch.ops.aten.where.self
                        for node in graph.graph.nodes
                    )
                )

    def test_combined_backward_requires_same_score_source(self):
        for same_source in (True, False):
            with self.subTest(same_source=same_source):

                def joint(scores, other_scores, bias, grad_weights, grad_per_expert):
                    indices = _selection(scores, bias)
                    weights, selected, denominator = _weights(scores, indices)
                    normalized, raw_norm, expanded_norm = _aux_forward(scores)
                    gate_scores = (
                        torch.ops.aten.alias.default(scores)
                        if same_source
                        else other_scores
                    )
                    zero = torch.ops.aten.scalar_tensor.default(
                        0.0, dtype=torch.float32, device=scores.device
                    )
                    pieces = _router_backward(
                        grad_weights,
                        selected,
                        denominator,
                        indices,
                        grad_per_expert,
                        scores,
                        raw_norm,
                        expanded_norm,
                        zero,
                        gate_scores,
                    )
                    return weights, normalized, pieces

                mode = FakeTensorMode()
                inputs = (
                    _fake_input(mode, (4096, 256)),
                    _fake_input(mode, (4096, 256)),
                    _fake_input(mode, (256,)),
                    _fake_input(mode, (4096, 8)),
                    _fake_input(mode, (256,)),
                )
                with mode:
                    graph = make_fx(joint)(*inputs)
                    counts = fuse_production_moe(
                        graph, regions=("selection", "weights", "gate_backward", "aux")
                    )
                self.assertEqual(counts["selection"], 1)
                self.assertEqual(counts["weights_forward"], 1)
                self.assertEqual(counts["aux_forward"], 1)
                self.assertEqual(counts.get("router_backward", 0), int(same_source))
                self.assertEqual(
                    counts.get("weights_gate_backward", 0), int(not same_source)
                )
                self.assertEqual(counts.get("aux_backward", 0), int(not same_source))


if __name__ == "__main__":
    unittest.main()
