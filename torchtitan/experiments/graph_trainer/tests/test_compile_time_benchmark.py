# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
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
            gm: torch.fx.GraphModule,
            selection: BenchmarkCandidateSelection,
            benchmark_regions: list[RewriteBenchmarkRegion] | None,
        ) -> torch.fx.GraphModule:
            applied = gm.meta.get("applied", ())
            for name in ("first", "second"):
                if name in applied or name in selection.rejected:
                    continue
                if selection.selected is not None and selection.selected != name:
                    continue
                selection.selected = name
                if benchmark_regions is None:
                    gm.meta["applied"] = (*applied, name)
                else:
                    benchmark_regions.append(regions[name])
                return gm
            return gm

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
                name="test",
                apply_candidate=apply_candidate,
                cache_key="test",
                benchmark_region=benchmark_region,
            )

        self.assertEqual(benchmark_region.call_count, 2)
        self.assertEqual(result.meta["applied"], ("second",))

    def test_accepted_rewrite_mutates_original_graph_module(self):
        constant = torch.randn(8)
        original = make_fx(lambda x: x + constant)(torch.randn(8))
        get_attr = next(node for node in original.graph.nodes if node.op == "get_attr")
        original_value = getattr(original, get_attr.target)

        def apply_candidate(
            gm: torch.fx.GraphModule,
            selection: BenchmarkCandidateSelection,
            benchmark_regions: list[RewriteBenchmarkRegion] | None,
        ) -> torch.fx.GraphModule:
            if gm.meta.get("applied") or selection.rejected:
                return gm
            selection.selected = "candidate"
            if benchmark_regions is None:
                gm.meta["applied"] = True
            else:
                benchmark_regions.append(
                    make_rewrite_benchmark_region(
                        make_fx(lambda x: x.neg())(torch.randn(8)),
                        make_fx(lambda x: x.relu())(torch.randn(8)),
                    )
                )
            return gm

        clear_compile_time_benchmark_cache()
        benchmark_region = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))
        with mock.patch("torch.cuda.is_available", return_value=True):
            result = apply_benchmarked_rewrites(
                original,
                name="test",
                apply_candidate=apply_candidate,
                cache_key="test",
                benchmark_region=benchmark_region,
            )

        self.assertTrue(result.meta["applied"])
        self.assertIs(result, original)
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
                cache_key="pattern",
                benchmark_region=benchmark,
            )
            second = benchmarker.benchmark_rewrite(
                graph,
                graph,
                cache_key="pattern",
                benchmark_region=benchmark,
            )

        benchmark.assert_called_once()
        self.assertFalse(first[0].cache_hit)
        self.assertTrue(second[0].cache_hit)

    def test_acceptance_policy_is_separate_from_measurement(self):
        benchmarker = CompileTimeBenchmarker(minimum_speedup=1.02)
        self.assertFalse(benchmarker.accepts(CompileTimeBenchmarkResult(1.0, 0.99)))
        self.assertTrue(benchmarker.accepts(CompileTimeBenchmarkResult(1.0, 0.95)))

    def test_benchmark_region_processes_baseline_and_candidate(self):
        output = torch.ones(2)
        baseline = mock.Mock()
        candidate = mock.Mock()
        processed_baseline = mock.Mock(return_value=output)
        processed_candidate = mock.Mock(return_value=output.clone())
        process_baseline = mock.Mock(return_value=processed_baseline)
        process_candidate = mock.Mock(return_value=processed_candidate)
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
        ):
            result = benchmarker.benchmark_region(
                baseline,
                (),
                candidate,
                (),
                process_baseline=process_baseline,
                process_candidate=process_candidate,
            )

        process_baseline.assert_called_once_with(baseline, (baseline_input,))
        process_candidate.assert_called_once_with(candidate, (candidate_input,))
        self.assertEqual(processed_baseline.call_count, 2)
        self.assertEqual(processed_candidate.call_count, 2)
        processed_baseline.assert_called_with(baseline_input)
        processed_candidate.assert_called_with(candidate_input)
        self.assertEqual(measurements, [(20, "median"), (20, "median")])
        self.assertEqual(result, CompileTimeBenchmarkResult(2.0, 1.0))

    def test_benchmark_region_uses_graph_module_candidate_by_default(self):
        output = torch.ones(2)
        baseline = mock.Mock(return_value=output)
        candidate = mock.Mock(return_value=output.clone())
        input_tensor = mock.Mock(device=torch.device("cuda:0"))
        benchmarker = CompileTimeBenchmarker()

        with (
            mock.patch.object(
                compile_time_benchmark,
                "_realize_paired_inputs",
                return_value=((input_tensor,), (input_tensor,)),
            ),
            mock.patch.object(
                compile_time_benchmark,
                "do_bench",
                side_effect=(2.0, 1.0),
            ),
        ):
            result = benchmarker.benchmark_region(baseline, (), candidate, ())

        self.assertEqual(baseline.call_count, 1)
        self.assertEqual(candidate.call_count, 1)
        self.assertEqual(result, CompileTimeBenchmarkResult(2.0, 1.0))

    def test_benchmark_region_rejects_incorrect_output(self):
        baseline = mock.Mock(return_value=torch.ones(2))
        candidate = mock.Mock()
        processed_candidate = mock.Mock(return_value=torch.zeros(2))
        input_tensor = mock.Mock(device=torch.device("cuda:0"))
        benchmarker = CompileTimeBenchmarker()

        with (
            mock.patch.object(
                compile_time_benchmark,
                "_realize_paired_inputs",
                return_value=((input_tensor,), (input_tensor,)),
            ),
            mock.patch.object(compile_time_benchmark, "do_bench") as do_bench,
            self.assertRaises(AssertionError),
        ):
            benchmarker.benchmark_region(
                baseline,
                (),
                candidate,
                (),
                process_candidate=lambda _candidate, _inputs: processed_candidate,
            )

        do_bench.assert_not_called()


if __name__ == "__main__":
    run_tests()
