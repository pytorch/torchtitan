# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.testing._internal.common_utils import TestCase

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    _QUANTIZATION_KIND,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _copy_fwd_metadata_to_bw_nodes,
)
from torchtitan.experiments.graph_trainer.configs import (
    FP8GraphConfig,
    GraphTrainerCompileConfig,
    validate_fp8_graph_config,
)
from torchtitan.experiments.graph_trainer.fp8_passes import (
    FP8_GEMM_TARGETS,
    analyze_fp8_regions_pass,
)
from torchtitan.experiments.graph_trainer.passes import compile_time_passes


class TestFP8GraphConfig(TestCase):
    def test_phase_one_contract(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="full",
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )
        validate_fp8_graph_config(config)

    def test_fp8_requires_full_inductor(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )
        with self.assertRaisesRegex(ValueError, "inductor_compilation full"):
            validate_fp8_graph_config(config)

    def test_fp8_requires_graph_passes(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            enable_passes=False,
            inductor_compilation="full",
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )
        with self.assertRaisesRegex(ValueError, "enable_passes"):
            validate_fp8_graph_config(config)

    def test_fp8_requires_cudagraph_to_be_disabled(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="full",
            fp8=FP8GraphConfig(enabled=True),
        )
        with self.assertRaisesRegex(ValueError, "cudagraph_pass"):
            validate_fp8_graph_config(config)

    def test_fp8_rejects_phase_two_precompile(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="full",
            disable_passes=["cudagraph_pass"],
            precompile_artifact_dir="/tmp/fp8",
            fp8=FP8GraphConfig(enabled=True),
        )
        with self.assertRaisesRegex(ValueError, "precompile_artifact_dir"):
            validate_fp8_graph_config(config)


class TestFP8Provenance(TestCase):
    def test_module_annotation_includes_quantization_kind(self) -> None:
        model = nn.Sequential(nn.ReLU())
        with patch(
            "torchtitan.components.quantization.utils.get_quantization_kind",
            return_value="float8_linear",
        ):
            annotate_module_fqns(model)

        with preserve_node_meta():
            gm = make_fx(model)(torch.randn(2, 2))
        custom_metadata = [
            node.meta.get("custom", {})
            for node in gm.graph.nodes
            if node.meta.get("custom", {}).get(_MODULE_FQN) == "0"
        ]
        self.assertTrue(custom_metadata)
        self.assertTrue(
            all(
                metadata[_QUANTIZATION_KIND] == "float8_linear"
                for metadata in custom_metadata
            )
        )

    def test_quantization_kind_is_copied_to_backward_nodes(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        forward = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        forward.meta["seq_nr"] = 1
        forward.meta["custom"] = {
            _MODULE_FQN: "layers.0.feed_forward.w1",
            _QUANTIZATION_KIND: "float8_linear",
        }
        backward = graph.call_function(torch.ops.aten.relu.default, args=(forward,))
        backward.meta["seq_nr"] = 1
        backward.meta["autograd_backward"] = True
        graph.output(backward)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        _copy_fwd_metadata_to_bw_nodes(gm)

        self.assertEqual(backward.meta["custom"], forward.meta["custom"])


class TestFP8AnalysisPass(TestCase):
    def _graph_with_quantized_node(self, target, *, backward: bool = False):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(target, args=(x,))
        node.meta["custom"] = {
            _MODULE_FQN: "layers.0.feed_forward.w1",
            _QUANTIZATION_KIND: "float8_linear",
        }
        if backward:
            node.meta["autograd_backward"] = True
        graph.output(node)
        return torch.fx.GraphModule(torch.nn.Module(), graph), node

    def test_records_forward_and_backward_scaled_gemms(self) -> None:
        self.assertTrue(FP8_GEMM_TARGETS)
        target = next(iter(FP8_GEMM_TARGETS))
        gm, forward = self._graph_with_quantized_node(target)
        output = next(node for node in gm.graph.nodes if node.op == "output")
        with gm.graph.inserting_before(output):
            backward = gm.graph.call_function(target, args=(forward,))
        backward.meta["custom"] = dict(forward.meta["custom"])
        backward.meta["autograd_backward"] = True
        output.args = (backward,)
        gm.recompile()

        analyze_fp8_regions_pass(gm, strict=True)

        region = gm.meta["fp8_summary"]["regions"][
            "('layers.0.feed_forward.w1', 'float8_linear')"
        ]
        self.assertEqual(region, {"forward_gemms": 1, "backward_gemms": 1})
        self.assertEqual(forward.meta["custom"]["fp8"]["op_role"], "gemm")
        self.assertEqual(backward.meta["custom"]["fp8"]["op_role"], "gemm")

    def test_strict_validation_rejects_missing_scaled_gemm(self) -> None:
        gm, _ = self._graph_with_quantized_node(torch.ops.aten.relu.default)

        with self.assertRaisesRegex(RuntimeError, "without a recognized FP8 GEMM"):
            analyze_fp8_regions_pass(gm, strict=True)

    def test_non_quantized_graph_is_a_noop(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        analyze_fp8_regions_pass(gm, strict=False)

        self.assertEqual(gm.meta["fp8_summary"], {"regions": {}, "target_inventory": {}})

    def test_strict_validation_rejects_missing_quantized_region(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with self.assertRaisesRegex(RuntimeError, "did not find quantized module"):
            analyze_fp8_regions_pass(gm, strict=True)


class TestFP8PassOrdering(TestCase):
    def test_analysis_follows_canonicalization(self) -> None:
        config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                inductor_compilation="full",
                disable_passes=["cudagraph_pass"],
                fp8=FP8GraphConfig(enabled=True),
            ),
            loss=CrossEntropyLoss.Config(),
            model_spec=SimpleNamespace(model=SimpleNamespace(layers=[0])),
            parallelism=SimpleNamespace(enable_async_tensor_parallel=False),
        )
        traced_result = SimpleNamespace(state_fqns=[])

        passes = compile_time_passes(traced_result, config)
        pass_names = [
            pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            for pass_fn in passes
        ]

        self.assertLess(
            pass_names.index("canonicalize_graph_pass"),
            pass_names.index("analyze_fp8_regions_pass"),
        )
