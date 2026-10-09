# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from unittest import mock

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer import compile_time_benchmark
from torchtitan.experiments.graph_trainer.compile_time_benchmark import (
    apply_benchmarked_rewrites,
    clear_compile_time_benchmark_cache,
    CompileTimeBenchmarker,
    CompileTimeBenchmarkResult,
    make_rewrite_benchmark_region,
    RewriteBenchmarkRegion,
    RewriteCandidate,
)


aten = torch.ops.aten


@torch.library.custom_op(
    "torchtitan_compile_time_benchmark_test::fast_mm",
    mutates_args=(),
)
def _fast_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.mm(lhs, rhs)


@_fast_mm.register_fake
def _fast_mm_fake(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return lhs.new_empty((lhs.shape[0], rhs.shape[1]))


fast_mm = torch.ops.torchtitan_compile_time_benchmark_test.fast_mm.default


class TestCompileTimeBenchmark(TestCase):
    def test_fast_mm_candidates_are_prepared_once(self):
        inputs = (
            torch.randn(2, 3),
            torch.randn(3, 4),
            torch.randn(4, 5),
        )
        original = make_fx(lambda x, w1, w2: torch.mm(torch.mm(x, w1), w2))(*inputs)
        expected = original(*inputs)

        def prepare_fast_mm_candidates(
            gm: torch.fx.GraphModule,
        ) -> list[RewriteCandidate]:
            candidates = []
            for mm in list(gm.graph.nodes):
                if mm.op != "call_function" or mm.target is not aten.mm.default:
                    continue
                baseline, _, _ = fuse_as_graphmodule(
                    gm,
                    [mm],
                    "FastMMBaseline",
                    always_return_tuple=True,
                )
                replacement = copy.deepcopy(baseline)
                replacement_mm = next(
                    node
                    for node in replacement.graph.nodes
                    if node.target is aten.mm.default
                )
                replacement_mm.target = fast_mm
                replacement.graph.lint()
                replacement.recompile()

                def apply(mm=mm):
                    with gm.graph.inserting_before(mm):
                        new_mm = gm.graph.call_function(fast_mm, mm.args, mm.kwargs)
                    new_mm.meta = mm.meta.copy()
                    mm.replace_all_uses_with(new_mm)
                    gm.graph.erase_node(mm)

                candidates.append(
                    RewriteCandidate(
                        f"fast_mm:{mm.name}",
                        (make_rewrite_benchmark_region(baseline, replacement),),
                        apply,
                    )
                )
            return candidates

        prepare_candidates = mock.Mock(side_effect=prepare_fast_mm_candidates)
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
                name="fast_mm",
                prepare_candidates=prepare_candidates,
                cache_key="fast_mm_v1",
                benchmark_region=benchmark_region,
            )

        result.graph.lint()
        result.recompile()
        targets = [node.target for node in result.graph.nodes]
        self.assertIs(result, original)
        prepare_candidates.assert_called_once_with(original)
        self.assertEqual(benchmark_region.call_count, 2)
        self.assertEqual(targets.count(aten.mm.default), 1)
        self.assertEqual(targets.count(fast_mm), 1)
        torch.testing.assert_close(result(*inputs), expected)

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

        with mock.patch.object(
            compile_time_benchmark,
            "_runtime_fingerprint",
            return_value=("runtime",),
        ):
            first = benchmarker.benchmark_regions(
                (region,),
                cache_key="pattern",
                benchmark_region=benchmark,
            )
            second = benchmarker.benchmark_regions(
                (region,),
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

    def test_numerical_tolerances_must_be_set_together(self):
        with self.assertRaisesRegex(ValueError, "both be specified or both omitted"):
            CompileTimeBenchmarker(atol=0.15)

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
        baseline = mock.Mock(return_value=torch.full((2,), 0.1))
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

    def test_benchmark_region_accepts_custom_tolerances(self):
        baseline = mock.Mock(return_value=torch.full((2,), 0.1))
        candidate = mock.Mock(return_value=torch.zeros(2))
        input_tensor = mock.Mock(device=torch.device("cuda:0"))
        benchmarker = CompileTimeBenchmarker(atol=0.15, rtol=0.05)

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

        self.assertEqual(result, CompileTimeBenchmarkResult(2.0, 1.0))


if __name__ == "__main__":
    run_tests()
