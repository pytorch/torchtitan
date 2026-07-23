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
from torchtitan.components.quantization.utils import QuantizationSignature
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    _QUANTIZATION_KIND,
    annotate_module_fqns,
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
    FP8_GEMM_TARGETS,
    annotate_fp8_for_regional_inductor_pass,
    validate_fp8_graph_pass,
)
from torchtitan.experiments.graph_trainer.inductor_passes import (
    regional_inductor_pass,
)
from torchtitan.experiments.graph_trainer.passes import compile_time_passes
from torchtitan.models.common.linear import Linear
from torchtitan.tools.utils import has_cuda_capability


class TestFP8GraphConfig(TestCase):
    def test_phase_one_contract(self) -> None:
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

    def test_fp8_requires_cudagraph_to_be_disabled(self) -> None:
        config = GraphTrainerCompileConfig(
            enable=True,
            inductor_compilation="full",
            fp8=FP8GraphConfig(enabled=True),
        )
        with self.assertRaisesRegex(ValueError, "cudagraph_pass"):
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

        validate_fp8_graph_pass(gm, strict=True)

        region = gm.meta["fp8_summary"]["regions"][
            "('layers.0.feed_forward.w1', 'float8_linear')"
        ]
        self.assertEqual(region, {"forward_gemms": 1, "backward_gemms": 1})
        self.assertNotIn("fp8", forward.meta["custom"])
        self.assertNotIn("fp8", backward.meta["custom"])
        self.assertNotIn("compile_with_inductor", forward.meta["custom"])
        self.assertNotIn("compile_with_inductor", backward.meta["custom"])

    def test_strict_validation_rejects_missing_scaled_gemm(self) -> None:
        gm, _ = self._graph_with_quantized_node(torch.ops.aten.relu.default)

        with self.assertRaisesRegex(RuntimeError, "without a recognized FP8 GEMM"):
            validate_fp8_graph_pass(gm, strict=True)

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

    def test_tags_connected_fp8_component(self) -> None:
        self.assertTrue(FP8_GEMM_TARGETS)
        target = next(iter(FP8_GEMM_TARGETS))
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        cast = self._node(graph, torch.ops.aten.clone.default, (x,))
        gemm = self._node(graph, target, (cast,))
        gemm.meta["custom"]["fp8"]["op_role"] = "gemm"
        graph.output(gemm)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with patch(
            "torchtitan.experiments.graph_trainer.fp8_passes._is_regional_fp8_compute_node",
            return_value=True,
        ):
            annotate_fp8_for_regional_inductor_pass(gm, strict=False)

        self.assertEqual(gm.meta["fp8_regional_summary"]["num_regions"], 1)
        self.assertEqual(cast.meta["custom"]["compile_with_inductor"], {})
        self.assertEqual(gemm.meta["custom"]["compile_with_inductor"], {})

    def test_grouped_experts_are_rejected(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = self._node(graph, torch.ops.aten.relu.default, (x,))
        node.meta["custom"][_QUANTIZATION_KIND] = "float8_grouped_experts"
        graph.output(node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with self.assertRaisesRegex(ValueError, "grouped experts"):
            annotate_fp8_for_regional_inductor_pass(gm, strict=False)


@unittest.skipUnless(
    torch.cuda.is_available()
    and has_cuda_capability(9, 0)
    and Float8Linear is not None,
    "FP8 regional compilation requires TorchAO and an H100-class GPU",
)
class TestFP8RegionalCompilation(TestCase):
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
            any(node.target in FP8_GEMM_TARGETS for node in traced.gm.graph.nodes)
        )

        annotate_fp8_for_regional_inductor_pass(traced.gm, strict=True)
        traced.gm = regional_inductor_pass(traced.gm, traced.example_inputs)
        actual = run_traced(traced, module=model)(input_tensor)

        self.assertEqual(len(actual), len(expected))
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)


@dataclass
class _FingerprintParallelDims:
    tp: int = 1


class TestFP8PrecompileFingerprint(TestCase):
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


class TestFP8PassOrdering(TestCase):
    def test_full_compilation_uses_validation_without_regional_annotation(self) -> None:
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

        self.assertIn("validate_fp8_graph_pass", pass_names)
        self.assertNotIn("annotate_fp8_for_regional_inductor_pass", pass_names)

    def test_regional_annotation_follows_graph_rewrites(self) -> None:
        config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                inductor_compilation="regional",
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
            pass_names.index("joint_transformer_block_bucketing_reordering_pass"),
            pass_names.index("annotate_fp8_for_regional_inductor_pass"),
        )
        self.assertLess(
            pass_names.index("annotate_fp8_for_regional_inductor_pass"),
            pass_names.index("annotate_flex_attention_for_regional_inductor_pass"),
        )
