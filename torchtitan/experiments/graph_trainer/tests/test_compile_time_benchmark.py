# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
from pathlib import Path
from unittest import mock

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer import compile_time_benchmark
from torchtitan.experiments.graph_trainer.compile_time_benchmark import (
    apply_benchmarked_rewrites,
    BenchmarkCandidateSelection,
    changed_nodes,
    clear_compile_time_benchmark_cache,
    CompileTimeBenchmarker,
    CompileTimeBenchmarkResult,
    extract_regions,
    make_rewrite_benchmark_region,
    RewriteBenchmarkRegion,
)


aten = torch.ops.aten


class _Diamond(torch.nn.Module):
    def forward(self, x):
        neg = torch.neg(x)
        return neg + torch.relu(neg)


class TestCompileTimeBenchmark(TestCase):
    def test_large_benchmark_summary_compacts_equivalent_results(self):
        applications = [
            compile_time_benchmark._BenchmarkApplication(
                f"pattern:candidate_{index}",
                "applied",
                (CompileTimeBenchmarkResult(2.0, 1.0, cache_hit=index > 0),),
            )
            for index in range(21)
        ]

        with (
            mock.patch.object(compile_time_benchmark.logger, "info") as log,
            mock.patch.object(compile_time_benchmark, "trace_structured") as trace,
        ):
            compile_time_benchmark._log_benchmark_summary(
                report_title="benchmark",
                artifact_name="benchmark_artifact",
                candidate_prefix="pattern:",
                candidate_label="FlexGEMM",
                applications=applications,
            )

        self.assertIn(
            "candidate candidate_0 (+20 equivalent)", log.call_args.args[0]
        )
        self.assertIn("cache=20/21 hits", log.call_args.args[0])
        full_report = trace.call_args.kwargs["payload_fn"]()
        self.assertIn("candidate candidate_0", full_report)
        self.assertIn("candidate candidate_20", full_report)

    def test_batch_rejects_candidate_without_benchmark_region(self):
        original = make_fx(lambda x: x.neg())(torch.randn(8))
        region_graph = make_fx(lambda x: x.relu())(torch.randn(8))
        placeholder = next(
            node for node in region_graph.graph.nodes if node.op == "placeholder"
        )
        signature = (((8,), (1,), torch.float32, "cpu", None),)
        region = RewriteBenchmarkRegion(
            region_graph,
            (placeholder,),
            region_graph,
            (placeholder,),
            (signature, signature),
        )

        def apply_candidate(current, selection, benchmark_regions):
            candidate = copy.deepcopy(current)
            names = ("valid", "missing")
            if selection.collect_all:
                for name in names:
                    selection.selected = name
                    selection.candidates.append(name)
                    if name == "valid":
                        benchmark_regions.extend((region,))
                candidate.meta["applied"] = names
            elif selection.accepted is not None:
                candidate.meta["applied"] = tuple(
                    name for name in names if name in selection.accepted
                )
            else:
                raise AssertionError("batch path was not used")
            return candidate

        benchmark = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))
        with mock.patch("torch.cuda.is_available", return_value=True):
            result = apply_benchmarked_rewrites(
                original,
                rewrite_name="test",
                apply_candidate=apply_candidate,
                namespace="test",
                benchmark_region=benchmark,
                batch_candidates=True,
            )

        benchmark.assert_called_once()
        self.assertEqual(result.meta["applied"], ("valid",))

    def test_applies_rewrite_candidates_independently(self):
        original = make_fx(lambda x: x.neg())(torch.randn(8))
        regions = {
            name: make_rewrite_benchmark_region(
                make_fx(lambda x: x.neg())(torch.randn(size)),
                make_fx(lambda x: x.relu())(torch.randn(size)),
            )
            for name, size in (("first", 8), ("second", 4))
        }

        def apply_candidate(
            current: torch.fx.GraphModule,
            selection: BenchmarkCandidateSelection,
            benchmark_regions: list[RewriteBenchmarkRegion],
        ) -> torch.fx.GraphModule:
            applied = current.meta.get("applied", ())
            for name in ("first", "second"):
                if name in applied or name in selection.rejected:
                    continue
                selection.selected = name
                benchmark_regions.append(regions[name])
                candidate = copy.deepcopy(current)
                candidate.meta["applied"] = (*applied, name)
                return candidate
            return current

        benchmark_region = mock.Mock(
            side_effect=(
                CompileTimeBenchmarkResult(1.0, 2.0),
                CompileTimeBenchmarkResult(2.0, 1.0),
            )
        )
        clear_compile_time_benchmark_cache()
        with mock.patch("torch.cuda.is_available", return_value=True):
            result = apply_benchmarked_rewrites(
                original,
                rewrite_name="test",
                apply_candidate=apply_candidate,
                namespace="test",
                benchmark_region=benchmark_region,
            )

        self.assertEqual(benchmark_region.call_count, 2)
        self.assertEqual(result.meta["applied"], ("second",))

    def test_accepted_rewrite_preserves_get_attr_identity(self):
        constant = torch.randn(8)
        original = make_fx(lambda x: x + constant)(torch.randn(8))
        get_attr = next(node for node in original.graph.nodes if node.op == "get_attr")
        original_value = getattr(original, get_attr.target)

        def apply_candidate(
            current: torch.fx.GraphModule,
            selection: BenchmarkCandidateSelection,
            benchmark_regions: list[RewriteBenchmarkRegion],
        ) -> torch.fx.GraphModule:
            if current.meta.get("applied") or selection.rejected:
                return current
            selection.selected = "candidate"
            benchmark_regions.append(
                make_rewrite_benchmark_region(
                    make_fx(lambda x: x.neg())(torch.randn(8)),
                    make_fx(lambda x: x.relu())(torch.randn(8)),
                )
            )
            candidate = copy.deepcopy(current)
            candidate.meta["applied"] = True
            return candidate

        clear_compile_time_benchmark_cache()
        benchmark_region = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))
        with mock.patch("torch.cuda.is_available", return_value=True):
            result = apply_benchmarked_rewrites(
                original,
                rewrite_name="test",
                apply_candidate=apply_candidate,
                namespace="test",
                benchmark_region=benchmark_region,
            )

        self.assertTrue(result.meta["applied"])
        self.assertIs(getattr(result, get_attr.target), original_value)

    def test_explicit_region_requires_matching_tensor_interfaces(self):
        baseline = make_fx(lambda x: x.neg())(torch.randn(8))
        matching = make_fx(lambda x: x.relu())(torch.randn(8))
        mismatched = make_fx(lambda x: x.relu())(torch.randn(4))
        ordered = make_fx(lambda x, y: x + y)(torch.randn(2, 1), torch.randn(1, 2))
        reordered = make_fx(lambda x, y: x + y)(torch.randn(1, 2), torch.randn(2, 1))

        region = make_rewrite_benchmark_region(baseline, matching)

        self.assertEqual(region.signature[0], region.signature[1])
        with self.assertRaisesRegex(RuntimeError, "signatures differ"):
            make_rewrite_benchmark_region(baseline, mismatched)
        with self.assertRaisesRegex(RuntimeError, "signatures differ"):
            make_rewrite_benchmark_region(ordered, reordered)

    def test_changed_regions_include_rewired_nodes_and_are_convex(self):
        baseline = make_fx(_Diamond())(torch.randn(8))
        candidate = copy.deepcopy(baseline)
        placeholder = next(
            node for node in candidate.graph.nodes if node.op == "placeholder"
        )
        old_neg = next(
            node for node in candidate.graph.nodes if node.target is aten.neg.default
        )
        relu = next(
            node for node in candidate.graph.nodes if node.target is aten.relu.default
        )
        old_add = next(
            node for node in candidate.graph.nodes if node.target is aten.add.Tensor
        )
        with candidate.graph.inserting_after(placeholder):
            new_neg = candidate.graph.call_function(aten.neg.default, (placeholder,))
        new_neg.meta = old_neg.meta.copy()
        relu.replace_input_with(old_neg, new_neg)
        with candidate.graph.inserting_before(old_add):
            new_add = candidate.graph.call_function(aten.add.Tensor, (new_neg, relu))
        new_add.meta = old_add.meta.copy()
        old_add.replace_all_uses_with(new_add)
        candidate.graph.erase_node(old_add)
        candidate.graph.erase_node(old_neg)
        candidate.graph.lint()

        baseline_nodes, candidate_nodes = changed_nodes(baseline, candidate)

        self.assertIn("relu", {node.name for node in baseline_nodes})
        self.assertIn("relu", {node.name for node in candidate_nodes})
        extract_regions(baseline, baseline_nodes, "Baseline")
        extract_regions(candidate, candidate_nodes, "Candidate")

    def test_equivalent_rewrites_reuse_measurement(self):
        graph = make_fx(lambda x: x.neg())(torch.randn(8))
        placeholder = next(
            node for node in graph.graph.nodes if node.op == "placeholder"
        )
        signature = (((8,), (1,), torch.float32, "cpu", None),)
        region = RewriteBenchmarkRegion(
            graph,
            (placeholder,),
            graph,
            (placeholder,),
            (signature, signature),
        )
        benchmark = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))
        benchmarker = CompileTimeBenchmarker()

        with (
            mock.patch.object(
                compile_time_benchmark,
                "infer_rewrite_regions",
                return_value=(region,),
            ),
            mock.patch.object(
                compile_time_benchmark,
                "_runtime_fingerprint",
                return_value=("runtime",),
            ),
        ):
            first = benchmarker.benchmark_rewrite(
                graph,
                graph,
                namespace="pattern",
                benchmark_region=benchmark,
            )
            second = benchmarker.benchmark_rewrite(
                graph,
                graph,
                namespace="pattern",
                benchmark_region=benchmark,
            )

        benchmark.assert_called_once()
        self.assertFalse(first[0].cache_hit)
        self.assertTrue(second[0].cache_hit)

    def test_persistent_cache_reuses_measurement_across_benchmarkers(self):
        graph = make_fx(lambda x: x.neg())(torch.randn(8))
        placeholder = next(
            node for node in graph.graph.nodes if node.op == "placeholder"
        )
        signature = (((8,), (1,), torch.float32, "cpu", None),)
        region = RewriteBenchmarkRegion(
            graph,
            (placeholder,),
            graph,
            (placeholder,),
            (signature, signature),
        )
        benchmark = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.object(
                compile_time_benchmark,
                "_runtime_fingerprint",
                return_value=("runtime",),
            ),
        ):
            cache_path = Path(tmp_dir) / "benchmark-cache.json"
            first = CompileTimeBenchmarker(cache_path=cache_path).benchmark_regions(
                (region,), namespace="pattern", benchmark_region=benchmark
            )
            second = CompileTimeBenchmarker(cache_path=cache_path).benchmark_regions(
                (region,), namespace="pattern", benchmark_region=benchmark
            )

        benchmark.assert_called_once()
        self.assertFalse(first[0].cache_hit)
        self.assertTrue(second[0].cache_hit)

    def test_persistent_cache_key_ignores_cuda_device_ordinal(self):
        cache_key_0 = (((4, 8), (8, 1), torch.bfloat16, "cuda", 0),)
        cache_key_1 = (((4, 8), (8, 1), torch.bfloat16, "cuda", 1),)

        self.assertEqual(
            CompileTimeBenchmarker._persistent_key(cache_key_0),
            CompileTimeBenchmarker._persistent_key(cache_key_1),
        )
        self.assertNotEqual(
            CompileTimeBenchmarker._legacy_persistent_key(cache_key_0),
            CompileTimeBenchmarker._legacy_persistent_key(cache_key_1),
        )

    def test_acceptance_policy_is_separate_from_measurement(self):
        benchmarker = CompileTimeBenchmarker(minimum_speedup=1.02)
        self.assertFalse(benchmarker.accepts(CompileTimeBenchmarkResult(1.0, 0.99)))
        self.assertTrue(benchmarker.accepts(CompileTimeBenchmarkResult(1.0, 0.95)))

    def test_benchmark_region_compares_eager_original(self):
        output = torch.ones(2)
        baseline = mock.Mock(return_value=output)
        candidate = mock.Mock()
        compiled_candidate = mock.Mock(return_value=output.clone())
        baseline_input = mock.Mock(device=torch.device("cuda:0"))
        candidate_input = mock.Mock(device=torch.device("cuda:0"))
        measurements = []

        def do_bench(fn, *, rep, return_mode):
            fn()
            measurements.append((rep, return_mode))
            return 2.0 if len(measurements) == 1 else 1.0

        benchmarker = CompileTimeBenchmarker()
        with (
            mock.patch.object(
                compile_time_benchmark,
                "_realize_paired_inputs",
                return_value=((baseline_input,), (candidate_input,)),
            ),
            mock.patch.object(
                compile_time_benchmark,
                "do_bench",
                side_effect=do_bench,
            ),
            mock.patch.object(
                torch,
                "compile",
                return_value=compiled_candidate,
            ) as compile_mock,
            mock.patch.object(torch.cuda, "empty_cache"),
        ):
            result = benchmarker.benchmark_region(baseline, (), candidate, ())

        compile_mock.assert_called_once_with(
            candidate,
            backend="inductor",
            fullgraph=True,
            mode="max-autotune-no-cudagraphs",
        )
        self.assertEqual(baseline.call_count, 2)
        self.assertEqual(compiled_candidate.call_count, 2)
        baseline.assert_called_with(baseline_input)
        compiled_candidate.assert_called_with(candidate_input)
        self.assertEqual(measurements, [(20, "median"), (20, "median")])
        self.assertEqual(result, CompileTimeBenchmarkResult(2.0, 1.0))

    def test_benchmark_region_rejects_incorrect_output(self):
        baseline = mock.Mock(return_value=torch.ones(2))
        candidate = mock.Mock()
        compiled_candidate = mock.Mock(return_value=torch.zeros(2))
        input_tensor = mock.Mock(device=torch.device("cuda:0"))
        benchmarker = CompileTimeBenchmarker()

        with (
            mock.patch.object(
                compile_time_benchmark,
                "_realize_paired_inputs",
                return_value=((input_tensor,), (input_tensor,)),
            ),
            mock.patch.object(torch, "compile", return_value=compiled_candidate),
            mock.patch.object(compile_time_benchmark, "do_bench") as do_bench,
            mock.patch.object(torch.cuda, "empty_cache"),
            self.assertRaises(AssertionError),
        ):
            benchmarker.benchmark_region(baseline, (), candidate, ())

        do_bench.assert_not_called()


if __name__ == "__main__":
    run_tests()
