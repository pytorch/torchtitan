# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.testing._internal.common_utils import TestCase

from torchtitan.components.quantization import Float8Linear, Float8LinearConverter
from torchtitan.components.quantization.float8 import (
    _get_float8_grouped_experts_cls,
)
from torchtitan.components.quantization.utils import (
    QuantizationSignature,
    get_quantization_signature,
)
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    _QUANTIZATION_EMULATE,
    _QUANTIZATION_KIND,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.cudagraph import (
    CUDAGraphWrapper,
    cudagraph_pass,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _copy_fwd_metadata_to_bw_nodes,
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.configs import (
    FP8GraphConfig,
    GraphTrainerCompileConfig,
    validate_fp8_graph_config,
)
from torchtitan.experiments.graph_trainer.fp8_passes import (
    FP8_COMPUTE_TARGETS,
    annotate_complete_fp8_regions_for_regional_inductor_pass,
    annotate_fp8_regions_for_regional_inductor_pass,
    validate_fp8_graph_pass,
)
from torchtitan.experiments.graph_trainer.inductor_passes import (
    regional_inductor_pass,
)
from torchtitan.experiments.graph_trainer.passes import (
    compile_time_passes,
    construct_default_graph_passes,
    final_inductor_compile_passes,
    graph_pp_pre_partition_fp8_passes,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.tools.utils import has_cuda_capability


class TestFP8GraphConfig(TestCase):
    def test_enabled_config_contract(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="full",
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )
        validate_fp8_graph_config(config)

    def test_fp8_rejects_unknown_inductor_mode(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="unknown",
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )
        with self.assertRaisesRegex(ValueError, "inductor_compilation full or regional"):
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

    def test_fp8_allows_cudagraph(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="full",
            fp8=FP8GraphConfig(enabled=True),
        )
        validate_fp8_graph_config(config)

    def test_fp8_precompile_requires_regional_inductor(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="full",
            disable_passes=["cudagraph_pass"],
            precompile_artifact_dir="/tmp/fp8",
            fp8=FP8GraphConfig(enabled=True),
        )
        with self.assertRaisesRegex(ValueError, "precompile requires"):
            validate_fp8_graph_config(config)

    def test_regional_fp8_precompile_is_valid(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="regional",
            disable_passes=["cudagraph_pass"],
            precompile_artifact_dir="/tmp/fp8",
            fp8=FP8GraphConfig(enabled=True),
        )
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
        self.assertTrue(
            all(
                metadata[_QUANTIZATION_EMULATE] is False
                for metadata in custom_metadata
            )
        )

    def test_module_annotation_includes_emulation_mode(self) -> None:
        model = nn.Sequential(nn.ReLU())
        model[0]._torchtitan_quantization_emulate = True
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
                metadata[_QUANTIZATION_EMULATE] is True
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


class TestFP8ValidationPass(TestCase):
    def _graph_with_quantized_node(
        self,
        target,
        *,
        backward: bool = False,
        quantization_kind: str = "float8_linear",
        data_operand_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(1, dtype=torch.bfloat16)
        data_operand = graph.placeholder("data_operand")
        data_operand.meta["val"] = torch.empty(1, dtype=data_operand_dtype)
        scale_operand = graph.placeholder("scale_operand")
        scale_operand.meta["val"] = torch.empty(1, dtype=torch.float8_e4m3fn)
        node = graph.call_function(target, args=(x, data_operand, scale_operand))
        node.meta["custom"] = {
            _MODULE_FQN: "layers.0.feed_forward.w1",
            _QUANTIZATION_KIND: quantization_kind,
        }
        if backward:
            node.meta["autograd_backward"] = True
        graph.output(node)
        return torch.fx.GraphModule(torch.nn.Module(), graph), node

    def test_records_forward_and_backward_fp8_compute_ops(self) -> None:
        targets = FP8_COMPUTE_TARGETS["float8_linear"]
        self.assertTrue(targets)
        target = next(iter(targets))
        gm, forward = self._graph_with_quantized_node(target)
        output = next(node for node in gm.graph.nodes if node.op == "output")
        data_operand = next(
            node for node in gm.graph.nodes if node.name == "data_operand"
        )
        scale_operand = next(
            node for node in gm.graph.nodes if node.name == "scale_operand"
        )
        with gm.graph.inserting_before(output):
            backward = gm.graph.call_function(
                target, args=(forward, data_operand, scale_operand)
            )
        backward.meta["custom"] = dict(forward.meta["custom"])
        backward.meta["autograd_backward"] = True
        output.args = (backward,)
        gm.recompile()

        validate_fp8_graph_pass(gm, strict=True)

        region = gm.meta["fp8_summary"]["regions"][
            "('layers.0.feed_forward.w1', 'float8_linear')"
        ]
        self.assertEqual(
            region,
            {"forward_compute_ops": 1, "backward_compute_ops": 1},
        )
        self.assertNotIn("fp8", forward.meta["custom"])
        self.assertNotIn("fp8", backward.meta["custom"])
        self.assertNotIn("compile_with_inductor", forward.meta["custom"])
        self.assertNotIn("compile_with_inductor", backward.meta["custom"])

    def test_strict_validation_rejects_missing_fp8_compute_op(self) -> None:
        gm, _ = self._graph_with_quantized_node(torch.ops.aten.relu.default)

        with self.assertRaisesRegex(RuntimeError, "without a supported FP8 compute"):
            validate_fp8_graph_pass(gm, strict=True)

    def test_compute_targets_are_selected_by_quantization_kind(self) -> None:
        targets = FP8_COMPUTE_TARGETS["mxfp8_linear"]
        self.assertTrue(targets)
        gm, _ = self._graph_with_quantized_node(
            next(iter(targets)),
            quantization_kind="mxfp8_linear",
        )

        validate_fp8_graph_pass(gm, strict=True)

        region = gm.meta["fp8_summary"]["regions"][
            "('layers.0.feed_forward.w1', 'mxfp8_linear')"
        ]
        self.assertEqual(
            region,
            {"forward_compute_ops": 1, "backward_compute_ops": 0},
        )

    def test_grouped_compute_requires_scaled_grouped_mm(self) -> None:
        dense_targets = FP8_COMPUTE_TARGETS["float8_linear"]
        self.assertTrue(dense_targets)

        for quantization_kind in (
            "float8_grouped_experts",
            "mxfp8_grouped_experts",
        ):
            grouped_targets = FP8_COMPUTE_TARGETS[quantization_kind]
            self.assertTrue(grouped_targets)
            grouped_gm, _ = self._graph_with_quantized_node(
                next(iter(grouped_targets)),
                quantization_kind=quantization_kind,
            )
            validate_fp8_graph_pass(grouped_gm, strict=True)

            dense_gm, _ = self._graph_with_quantized_node(
                next(iter(dense_targets)),
                quantization_kind=quantization_kind,
            )
            with self.assertRaisesRegex(
                RuntimeError, "without a supported FP8 compute"
            ):
                validate_fp8_graph_pass(dense_gm, strict=True)

    def test_fp8_scale_operand_does_not_prove_fp8_compute(self) -> None:
        targets = FP8_COMPUTE_TARGETS["float8_linear"]
        self.assertTrue(targets)
        gm, _ = self._graph_with_quantized_node(
            next(iter(targets)),
            data_operand_dtype=torch.bfloat16,
        )

        with self.assertRaisesRegex(RuntimeError, "without a supported FP8 compute"):
            validate_fp8_graph_pass(gm, strict=True)

    def test_emulated_float8_region_does_not_require_scaled_compute(self) -> None:
        gm, node = self._graph_with_quantized_node(torch.ops.aten.mm.default)
        node.meta["custom"][_QUANTIZATION_EMULATE] = True

        validate_fp8_graph_pass(gm, strict=True)

        region = gm.meta["fp8_summary"]["regions"][
            "('layers.0.feed_forward.w1', 'float8_linear')"
        ]
        self.assertEqual(
            region,
            {"forward_compute_ops": 0, "backward_compute_ops": 0},
        )

    def test_non_quantized_graph_is_a_noop(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        validate_fp8_graph_pass(gm, strict=False)

        self.assertEqual(gm.meta["fp8_summary"], {"regions": {}, "target_inventory": {}})

    def test_strict_validation_rejects_missing_quantized_region(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with self.assertRaisesRegex(RuntimeError, "did not find quantized module"):
            validate_fp8_graph_pass(gm, strict=True)


class TestFP8RegionalAnnotation(TestCase):
    def _node(self, graph, target, args=()):
        node = graph.call_function(target, args=args)
        node.meta["val"] = torch.empty(1, device="meta")
        node.meta["custom"] = {
            _MODULE_FQN: "layers.0.feed_forward.w1",
            _QUANTIZATION_KIND: "float8_linear",
            "fp8": {"op_role": "cast"},
        }
        return node

    def test_identifies_and_tags_connected_fp8_component(self) -> None:
        targets = FP8_COMPUTE_TARGETS["float8_linear"]
        self.assertTrue(targets)
        target = next(iter(targets))
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        cast = self._node(graph, torch.ops.aten.clone.default, (x,))
        cast.meta["val"] = torch.empty(
            1, device="meta", dtype=torch.float8_e4m3fn
        )
        gemm = self._node(graph, target, (cast,))
        gemm.meta["custom"]["fp8"]["op_role"] = "compute"
        graph.output(gemm)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with patch(
            "torchtitan.experiments.graph_trainer.fp8_passes._is_regional_fp8_compute_node",
            return_value=True,
        ):
            annotate_fp8_regions_for_regional_inductor_pass(gm, strict=False)
        annotate_complete_fp8_regions_for_regional_inductor_pass(gm)

        self.assertEqual(gm.meta["fp8_regional_summary"]["num_regions"], 1)
        expected = {"inductor_region": 0}
        self.assertEqual(cast.meta["custom"]["compile_with_inductor"], expected)
        self.assertEqual(gemm.meta["custom"]["compile_with_inductor"], expected)

    def test_identifies_and_tags_grouped_experts_component(self) -> None:
        for quantization_kind in (
            "float8_grouped_experts",
            "mxfp8_grouped_experts",
        ):
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            cast = self._node(graph, torch.ops.aten.clone.default, (x,))
            compute = self._node(
                graph,
                next(iter(FP8_COMPUTE_TARGETS[quantization_kind])),
                (cast,),
            )
            for node in (cast, compute):
                node.meta["custom"][_MODULE_FQN] = (
                    "layers.0.moe.routed_experts.inner_experts"
                )
                node.meta["custom"][_QUANTIZATION_KIND] = quantization_kind
            cast.meta["val"] = torch.empty(
                1, device="meta", dtype=torch.float8_e4m3fn
            )
            compute.meta["custom"]["fp8"]["op_role"] = "compute"
            graph.output(compute)
            gm = torch.fx.GraphModule(torch.nn.Module(), graph)

            with patch(
                "torchtitan.experiments.graph_trainer.fp8_passes._is_regional_fp8_compute_node",
                return_value=True,
            ):
                annotate_fp8_regions_for_regional_inductor_pass(gm, strict=False)
            annotate_complete_fp8_regions_for_regional_inductor_pass(gm)

            self.assertEqual(gm.meta["fp8_regional_summary"]["num_regions"], 1)
            expected = {"inductor_region": 0}
            self.assertEqual(cast.meta["custom"]["compile_with_inductor"], expected)
            self.assertEqual(
                compute.meta["custom"]["compile_with_inductor"], expected
            )

    def test_identifies_fp8_region_without_tagging_inductor(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        cast = self._node(graph, torch.ops.aten.clone.default, (x,))
        cast.meta["val"] = torch.empty(
            1, device="meta", dtype=torch.float8_e4m3fn
        )
        compute = self._node(
            graph,
            next(iter(FP8_COMPUTE_TARGETS["float8_linear"])),
            (cast,),
        )
        graph.output(compute)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with patch(
            "torchtitan.experiments.graph_trainer.fp8_passes._is_regional_fp8_compute_node",
            return_value=True,
        ):
            annotate_fp8_regions_for_regional_inductor_pass(
                gm,
                strict=False,
            )

        self.assertNotIn("compile_with_inductor", cast.meta["custom"])
        self.assertNotIn("compile_with_inductor", compute.meta["custom"])
        self.assertEqual(cast.meta["custom"]["fp8"]["regional_region_num_nodes"], 2)

    def test_tags_only_complete_identified_fp8_region(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        cast = self._node(graph, torch.ops.aten.clone.default, (x,))
        compute = self._node(
            graph,
            next(iter(FP8_COMPUTE_TARGETS["float8_linear"])),
            (cast,),
        )
        for node in (cast, compute):
            node.meta["custom"]["fp8"].update(
                {"regional_region_id": 3, "regional_region_num_nodes": 2}
            )
        compute.meta["custom"]["fp8"]["op_role"] = "compute"
        other = graph.call_function(torch.ops.aten.relu.default, args=(compute,))
        other.meta["custom"] = {"compile_with_inductor": {"source": "flex"}}
        graph.output(other)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        annotate_complete_fp8_regions_for_regional_inductor_pass(gm)

        expected = {"inductor_region": 3}
        self.assertEqual(cast.meta["custom"]["compile_with_inductor"], expected)
        self.assertEqual(compute.meta["custom"]["compile_with_inductor"], expected)
        self.assertEqual(
            other.meta["custom"]["compile_with_inductor"],
            {"source": "flex"},
        )

        partial_graph = torch.fx.Graph()
        partial_input = partial_graph.placeholder("x")
        partial_cast = self._node(
            partial_graph,
            torch.ops.aten.clone.default,
            (partial_input,),
        )
        partial_cast.meta["custom"]["fp8"].update(
            {"regional_region_id": 3, "regional_region_num_nodes": 2}
        )
        partial_graph.output(partial_cast)
        partial_gm = torch.fx.GraphModule(torch.nn.Module(), partial_graph)

        with self.assertWarnsRegex(UserWarning, "incomplete FP8 regional"):
            annotate_complete_fp8_regions_for_regional_inductor_pass(partial_gm)

        self.assertNotIn("compile_with_inductor", partial_cast.meta["custom"])

    def test_reidentifies_partitioned_fp8_regions(self) -> None:
        target = next(iter(FP8_COMPUTE_TARGETS["float8_linear"]))

        def make_partition(region_id: int) -> tuple[torch.fx.GraphModule, list]:
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            cast = self._node(graph, torch.ops.aten.clone.default, (x,))
            cast.meta["val"] = torch.empty(
                1, device="meta", dtype=torch.float8_e4m3fn
            )
            compute = self._node(graph, target, (cast,))
            compute.meta["custom"]["fp8"]["op_role"] = "compute"
            graph.output(compute)
            for node in (cast, compute):
                node.meta["custom"]["fp8"].update(
                    {
                        "regional_region_id": region_id,
                        "regional_region_num_nodes": 4,
                    }
                )
            return torch.fx.GraphModule(torch.nn.Module(), graph), [cast, compute]

        for region_id in (0, 1):
            gm, nodes = make_partition(region_id)
            with patch(
                "torchtitan.experiments.graph_trainer.fp8_passes."
                "_is_regional_fp8_compute_node",
                return_value=True,
            ):
                annotate_fp8_regions_for_regional_inductor_pass(gm, strict=False)
            annotate_complete_fp8_regions_for_regional_inductor_pass(gm)

            self.assertEqual(gm.meta["fp8_regional_summary"]["num_regions"], 1)
            self.assertTrue(
                all(
                    node.meta["custom"]["fp8"]["regional_region_num_nodes"] == 2
                    for node in nodes
                )
            )
            self.assertTrue(
                all("compile_with_inductor" in node.meta["custom"] for node in nodes)
            )

    def test_tags_compute_after_graph_pp_fp8_input_boundary(self) -> None:
        target = next(iter(FP8_COMPUTE_TARGETS["float8_linear"]))
        graph = torch.fx.Graph()
        grad_output_fp8 = graph.placeholder("grad_output_fp8")
        weight_fp8 = graph.placeholder("weight_fp8")
        for node in (grad_output_fp8, weight_fp8):
            node.meta["val"] = torch.empty(
                1, device="meta", dtype=torch.float8_e4m3fn
            )
            node.meta["custom"] = {
                _MODULE_FQN: "layers.0.feed_forward.w1",
                _QUANTIZATION_KIND: "float8_linear",
            }
        compute = self._node(graph, target, (grad_output_fp8, weight_fp8))
        graph.output(compute)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with patch(
            "torchtitan.experiments.graph_trainer.fp8_passes."
            "_is_regional_fp8_compute_node",
            side_effect=lambda node, **_kwargs: node.op == "call_function",
        ):
            annotate_fp8_regions_for_regional_inductor_pass(gm, strict=False)
        annotate_complete_fp8_regions_for_regional_inductor_pass(gm)

        for boundary in (grad_output_fp8, weight_fp8):
            self.assertEqual(
                boundary.meta["custom"]["fp8"]["op_role"], "input_boundary"
            )
            self.assertNotIn("compile_with_inductor", boundary.meta["custom"])
        self.assertEqual(compute.meta["custom"]["fp8"]["op_role"], "compute")
        self.assertEqual(
            compute.meta["custom"]["fp8"]["regional_region_num_nodes"], 1
        )
        self.assertEqual(
            compute.meta["custom"]["compile_with_inductor"],
            {"inductor_region": 0},
        )


@unittest.skipUnless(
    torch.cuda.is_available()
    and has_cuda_capability(9, 0)
    and Float8Linear is not None,
    "FP8 regional compilation requires TorchAO and an H100-class GPU",
)
class TestFP8RegionalCompilation(TestCase):
    def test_float8_grouped_experts_regional_inductor_runs_forward_and_backward(
        self,
    ) -> None:
        float8_grouped_experts = _get_float8_grouped_experts_cls(GroupedExperts)
        model = float8_grouped_experts.Config(
            dim=16,
            hidden_dim=16,
            num_experts=2,
        ).build()
        model = model.to(device="cuda", dtype=torch.bfloat16)
        annotate_module_fqns(nn.Sequential(model))
        input_tensor = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
        num_tokens_per_expert = torch.tensor(
            [16, 16], device="cuda", dtype=torch.int64
        )

        def train_step(
            x: torch.Tensor, num_tokens: torch.Tensor
        ) -> list[torch.Tensor]:
            output = model(x, num_tokens)
            loss = output.float().sum()
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            return [loss, *grads]

        expected = train_step(input_tensor, num_tokens_per_expert)
        traced = minimal_fx_tracer(train_step, module=model)(
            input_tensor, num_tokens_per_expert
        )
        self.assertTrue(
            any(
                node.target in FP8_COMPUTE_TARGETS["float8_grouped_experts"]
                for node in traced.gm.graph.nodes
            )
        )

        annotate_fp8_regions_for_regional_inductor_pass(traced.gm, strict=True)
        annotate_complete_fp8_regions_for_regional_inductor_pass(traced.gm)
        traced.gm = regional_inductor_pass(traced.gm, traced.example_inputs)
        actual = run_traced(traced, module=model)(
            input_tensor, num_tokens_per_expert
        )

        self.assertEqual(len(actual), len(expected))
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_float8_linear_converter_regional_inductor_runs_forward_and_backward(
        self,
    ) -> None:
        converter = Float8LinearConverter(
            Float8LinearConverter.Config(model_compile_enabled=True)
        )
        linear_config = converter.convert(
            Linear.Config(in_features=16, out_features=16, bias=False)
        )
        self.assertIsInstance(linear_config, Float8Linear.Config)
        model = nn.Sequential(linear_config.build()).to(
            device="cuda", dtype=torch.bfloat16
        )
        annotate_module_fqns(model)
        input_tensor = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)

        def train_step(x: torch.Tensor) -> list[torch.Tensor]:
            output = model(x)
            loss = output.float().sum()
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            return [loss, *grads]

        expected = train_step(input_tensor)
        traced = minimal_fx_tracer(train_step, module=model)(input_tensor)
        self.assertTrue(
            any(
                node.target in FP8_COMPUTE_TARGETS["float8_linear"]
                for node in traced.gm.graph.nodes
            )
        )

        annotate_fp8_regions_for_regional_inductor_pass(traced.gm, strict=True)
        annotate_complete_fp8_regions_for_regional_inductor_pass(traced.gm)
        traced.gm = regional_inductor_pass(traced.gm, traced.example_inputs)
        actual = run_traced(traced, module=model)(input_tensor)

        self.assertEqual(len(actual), len(expected))
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_float8_regional_inductor_cudagraph_replays_dynamic_inputs_and_state(
        self,
    ) -> None:
        converter = Float8LinearConverter(
            Float8LinearConverter.Config(model_compile_enabled=True)
        )
        linear_config = converter.convert(
            Linear.Config(in_features=16, out_features=16, bias=False)
        )
        model = nn.Sequential(linear_config.build()).to(
            device="cuda", dtype=torch.bfloat16
        )
        annotate_module_fqns(model)

        def train_step(x: torch.Tensor) -> list[torch.Tensor]:
            output = model(x)
            loss = output.float().square().mean()
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            return [loss, *grads]

        inputs = [
            torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        traced = minimal_fx_tracer(train_step, module=model)(inputs[0])
        annotate_fp8_regions_for_regional_inductor_pass(traced.gm, strict=True)
        annotate_complete_fp8_regions_for_regional_inductor_pass(traced.gm)
        traced.gm = regional_inductor_pass(traced.gm, traced.example_inputs)
        traced.gm = cudagraph_pass(
            traced.gm,
            traced.example_inputs,
            static_input_indices=tuple(range(traced.num_static_inputs)),
            tensor_input_indices=traced.tensor_input_indices,
        )
        self.assertIsInstance(traced.gm.forward, CUDAGraphWrapper)
        compiled_step = run_traced(traced, module=model)

        # Call 1 warms up, call 2 captures, and call 3 replays with both new
        # activation values and updated parameter contents at stable addresses.
        for index, input_tensor in enumerate(inputs):
            expected = train_step(input_tensor)
            actual = compiled_step(input_tensor)
            self.assertEqual(len(actual), len(expected))
            for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
                torch.testing.assert_close(actual_tensor, expected_tensor)
            if index == 1:
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(0.125)
        self.assertIsNotNone(traced.gm.forward._cudagraph)


@dataclass
class _FingerprintParallelDims:
    tp: int = 1


class TestFP8PrecompileFingerprint(TestCase):
    def test_grouped_expert_signature_includes_padding_contract(self) -> None:
        grouped_experts = nn.Linear(2, 2)
        grouped_experts._torchtitan_quantization_recipe_name = "mxfp8_rceil"
        grouped_experts._torchtitan_quantization_emulate = False
        grouped_experts._torchtitan_quantization_pad_multiple = 128
        model = nn.Module()
        model.add_module("inner_experts", grouped_experts)

        with patch(
            "torchtitan.components.quantization.utils.get_quantization_kind",
            side_effect=lambda module: (
                "mxfp8_grouped_experts" if module is grouped_experts else None
            ),
        ):
            signatures = get_quantization_signature(model)

        self.assertEqual(
            signatures,
            (
                QuantizationSignature(
                    module_fqn="inner_experts",
                    kind="mxfp8_grouped_experts",
                    recipe_name="mxfp8_rceil",
                    emulate=False,
                    pad_multiple=128,
                ),
            ),
        )

    def test_quantization_signature_changes_fingerprint(self) -> None:
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="regional",
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )
        rowwise = QuantizationSignature(
            module_fqn="layers.0.w1",
            kind="float8_linear",
            recipe_name="rowwise",
            emulate=False,
        )
        rowwise_with_gw_hp = QuantizationSignature(
            module_fqn="layers.0.w1",
            kind="float8_linear",
            recipe_name="rowwise_with_gw_hp",
            emulate=False,
        )
        model = torch.nn.Module()
        dims = _FingerprintParallelDims()
        with patch(
            "torchtitan.experiments.graph_trainer.precompile.get_quantization_signature",
            return_value=(rowwise,),
        ):
            rowwise_fingerprint = compute_config_fingerprint(model, config, dims)
        with patch(
            "torchtitan.experiments.graph_trainer.precompile.get_quantization_signature",
            return_value=(rowwise_with_gw_hp,),
        ):
            gw_hp_fingerprint = compute_config_fingerprint(model, config, dims)

        self.assertNotEqual(rowwise_fingerprint, gw_hp_fingerprint)

    def test_grouped_expert_padding_changes_fingerprint(self) -> None:
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="regional",
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )
        signature = QuantizationSignature(
            module_fqn="layers.0.moe.routed_experts.inner_experts",
            kind="mxfp8_grouped_experts",
            recipe_name="mxfp8_rceil",
            emulate=False,
            pad_multiple=32,
        )
        model = torch.nn.Module()
        dims = _FingerprintParallelDims()
        with patch(
            "torchtitan.experiments.graph_trainer.precompile.get_quantization_signature",
            return_value=(signature,),
        ):
            pad_32_fingerprint = compute_config_fingerprint(model, config, dims)
        signature = QuantizationSignature(
            module_fqn=signature.module_fqn,
            kind=signature.kind,
            recipe_name=signature.recipe_name,
            emulate=signature.emulate,
            pad_multiple=128,
        )
        with patch(
            "torchtitan.experiments.graph_trainer.precompile.get_quantization_signature",
            return_value=(signature,),
        ):
            pad_128_fingerprint = compute_config_fingerprint(model, config, dims)

        self.assertNotEqual(pad_32_fingerprint, pad_128_fingerprint)

    def test_torchao_version_changes_fingerprint(self) -> None:
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="regional",
            fp8=FP8GraphConfig(enabled=True),
        )
        model = torch.nn.Module()
        dims = _FingerprintParallelDims()
        with (
            patch(
                "torchtitan.experiments.graph_trainer.precompile."
                "get_quantization_signature",
                return_value=(),
            ),
            patch(
                "torchtitan.experiments.graph_trainer.precompile.package_version",
                return_value="0.16.0",
            ),
        ):
            old_fingerprint = compute_config_fingerprint(model, config, dims)
        with (
            patch(
                "torchtitan.experiments.graph_trainer.precompile."
                "get_quantization_signature",
                return_value=(),
            ),
            patch(
                "torchtitan.experiments.graph_trainer.precompile.package_version",
                return_value="0.17.0",
            ),
        ):
            new_fingerprint = compute_config_fingerprint(model, config, dims)

        self.assertNotEqual(old_fingerprint, new_fingerprint)


class TestFP8PassOrdering(TestCase):
    def test_full_inductor_precedes_cudagraph(self) -> None:
        config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                inductor_compilation="full",
                fp8=FP8GraphConfig(enabled=True),
            ),
            loss=CrossEntropyLoss.Config(),
            model_spec=SimpleNamespace(model=SimpleNamespace(layers=[0])),
            parallelism=SimpleNamespace(enable_async_tensor_parallel=False),
        )
        traced_result = SimpleNamespace(
            state_fqns=[],
            num_static_inputs=1,
            tensor_input_indices=[0, 1],
        )

        passes = construct_default_graph_passes(traced_result, config)
        pass_names = [
            pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            for pass_fn in passes
        ]

        self.assertLess(
            pass_names.index("full_inductor_compilation_pass"),
            pass_names.index("cudagraph_pass"),
        )

    def test_full_terminal_fp8_pass_inclusion(self) -> None:
        config = GraphTrainerCompileConfig(
            inductor_compilation="full",
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )

        fp8_passes = final_inductor_compile_passes(config)
        fp8_pass_names = [
            pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            for pass_fn in fp8_passes
        ]
        self.assertEqual(
            fp8_pass_names,
            ["validate_fp8_graph_pass", "full_inductor_compilation_pass"],
        )

        config.fp8.enabled = False
        skipped_passes = final_inductor_compile_passes(config)
        self.assertEqual(len(skipped_passes), 1)

    def test_regional_terminal_fp8_pass_inclusion(self) -> None:
        config = GraphTrainerCompileConfig(
            inductor_compilation="regional",
            numerics_changing_optim=True,
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )

        passes = final_inductor_compile_passes(config)
        pass_names = [
            pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            for pass_fn in passes
        ]
        self.assertLess(
            pass_names.index("annotate_rmsnorm_for_regional_inductor_pass"),
            pass_names.index("annotate_fp8_regions_for_regional_inductor_pass"),
        )
        self.assertLess(
            pass_names.index("annotate_fp8_regions_for_regional_inductor_pass"),
            pass_names.index(
                "annotate_complete_fp8_regions_for_regional_inductor_pass"
            ),
        )
        self.assertLess(
            pass_names.index(
                "annotate_complete_fp8_regions_for_regional_inductor_pass"
            ),
            pass_names.index("regional_inductor_pass"),
        )

        config.fp8.enabled = False
        skipped_passes = final_inductor_compile_passes(config)
        skipped_names = [
            pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            for pass_fn in skipped_passes
        ]
        self.assertNotIn(
            "annotate_fp8_regions_for_regional_inductor_pass", skipped_names
        )
        self.assertNotIn(
            "annotate_complete_fp8_regions_for_regional_inductor_pass",
            skipped_names,
        )

    def test_full_validation_follows_graph_rewrites(self) -> None:
        config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                inductor_compilation="full",
                disable_passes=["cudagraph_pass"],
                fp8=FP8GraphConfig(enabled=True),
            ),
            loss=CrossEntropyLoss.Config(),
            model_spec=SimpleNamespace(model=SimpleNamespace(layers=[0])),
            parallelism=SimpleNamespace(enable_async_tensor_parallel=True),
        )
        traced_result = SimpleNamespace(state_fqns=[])

        passes = compile_time_passes(traced_result, config)
        pass_names = [
            pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            for pass_fn in passes
        ]

        self.assertIn("validate_fp8_graph_pass", pass_names)
        self.assertNotIn("annotate_fp8_regions_for_regional_inductor_pass", pass_names)
        self.assertLess(
            pass_names.index("async_tensor_parallel_pass"),
            pass_names.index("validate_fp8_graph_pass"),
        )
        self.assertLess(
            pass_names.index("validate_fp8_graph_pass"),
            pass_names.index("full_inductor_compilation_pass"),
        )

    def test_regional_annotation_precedes_regional_inductor(self) -> None:
        config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                inductor_compilation="regional",
                disable_passes=["cudagraph_pass"],
                numerics_changing_optim=True,
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
            pass_names.index("annotate_flex_attention_for_regional_inductor_pass"),
            pass_names.index("annotate_rmsnorm_for_regional_inductor_pass"),
        )
        self.assertLess(
            pass_names.index("annotate_rmsnorm_for_regional_inductor_pass"),
            pass_names.index("annotate_fp8_regions_for_regional_inductor_pass"),
        )
        self.assertLess(
            pass_names.index("annotate_fp8_regions_for_regional_inductor_pass"),
            pass_names.index("regional_inductor_pass"),
        )

    def test_graph_pp_validates_before_partitioning(self) -> None:
        compile_config = GraphTrainerCompileConfig(
            inductor_compilation="regional",
            disable_passes=["cudagraph_pass"],
            fp8=FP8GraphConfig(enabled=True),
        )

        passes = graph_pp_pre_partition_fp8_passes(compile_config)
        pass_names = [
            pass_fn.func.__name__ if hasattr(pass_fn, "func") else pass_fn.__name__
            for pass_fn in passes
        ]

        self.assertEqual(pass_names, ["validate_fp8_graph_pass"])
