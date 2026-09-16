# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the auto_perf_maxing solver's pure helpers.

Everything here runs without a GPU and without a distributed process group.
The solver end to end needs both; these cover the pieces that decide what it
plans, so a regression in the cost/size/budget arithmetic is caught in CI
rather than by an OOM on an 8-GPU job.
"""

import unittest
from dataclasses import dataclass, field
from unittest import mock

import torch

from torchtitan.experiments.graph_trainer.auto_sac_offload_solver import (
    _fmt_ranges,
    _fsdp_shard_degree,
    _validate_fractions,
    get_knapsack,
)
from torchtitan.experiments.graph_trainer.auto_sac_offload_solver_helpers import (
    local_param_numel,
    optimizer_state_bytes,
)
from torchtitan.experiments.graph_trainer.auto_sac_offload_solver_utils import (
    _concrete_bytes,
    get_size,
    ROUNDING,
    UNKNOWN_OPTIMIZER_BYTES,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    _touches_backward,
)
from torchtitan.experiments.graph_trainer.configs import AutoSacConfig
from torchtitan.experiments.graph_trainer.memory_policy import resolve_memory_budget_gib

MiB = 1 << 20


@dataclass
class _FakeParamGroup:
    optimizer_name: str = "AdamW"
    optimizer_kwargs: dict = field(default_factory=dict)


@dataclass
class _FakeOptConfig:
    param_groups: list = field(default_factory=lambda: [_FakeParamGroup()])
    implementation: str = "foreach"


class TestGetKnapsack(unittest.TestCase):
    """The inner greedy's exact 0/1 knapsack over a MiB-quantised size axis."""

    def _items(self, spec):
        """spec: {name: (size_mib, cost)} -> (names, size_of, cost_of)."""
        size = {n: s * MiB for n, (s, _) in spec.items()}
        cost = {n: c for n, (_, c) in spec.items()}
        return list(spec), size.get, cost.get

    def test_at_most_maximizes_value_within_limit(self):
        names, size_of, cost_of = self._items(
            {"a": (1, 10.0), "b": (2, 25.0), "c": (3, 30.0)}
        )
        # a+b is 3 MiB for 35.0; c alone is 3 MiB for 30.0. First-fit by
        # cost-per-byte would take c first and then stop.
        got = get_knapsack(names, 3 * MiB, size_of, cost_of, "at_most")
        self.assertEqual(sorted(got), ["a", "b"])

    def test_at_least_minimizes_cost_while_covering_limit(self):
        names, size_of, cost_of = self._items(
            {"a": (1, 10.0), "b": (2, 25.0), "c": (3, 30.0)}
        )
        got = get_knapsack(names, 3 * MiB, size_of, cost_of, "at_least")
        self.assertEqual(sorted(got), ["c"])

    def test_at_least_may_overshoot_to_cover(self):
        """No exact cover exists, so it must take the smallest sufficient set."""
        names, size_of, cost_of = self._items({"a": (4, 1.0), "b": (10, 5.0)})
        got = get_knapsack(names, 5 * MiB, size_of, cost_of, "at_least")
        self.assertEqual(got, ["b"])

    def test_at_least_returns_empty_when_unreachable(self):
        names, size_of, cost_of = self._items({"a": (1, 1.0), "b": (2, 1.0)})
        self.assertEqual(
            get_knapsack(names, 100 * MiB, size_of, cost_of, "at_least"), []
        )

    def test_nonpositive_limit_selects_nothing(self):
        names, size_of, cost_of = self._items({"a": (1, 1.0)})
        for limit in (0, -1, -(8 * MiB)):
            self.assertEqual(
                get_knapsack(names, limit, size_of, cost_of, "at_most"), []
            )

    def test_sub_unit_nodes_are_free_under_at_most_and_ignored_under_at_least(self):
        spec = {"tiny": (0, 1.0), "big": (2, 5.0)}
        size = {"tiny": 1024, "big": 2 * MiB}  # tiny is under one 1 MiB unit
        cost = {n: c for n, (_, c) in spec.items()}
        at_most = get_knapsack(list(spec), 2 * MiB, size.get, cost.get, "at_most")
        self.assertIn("tiny", at_most)
        at_least = get_knapsack(list(spec), 2 * MiB, size.get, cost.get, "at_least")
        self.assertNotIn("tiny", at_least)

    def test_empty_candidates(self):
        self.assertEqual(
            get_knapsack([], 4 * MiB, lambda n: 0, lambda n: 0.0, "at_most"), []
        )
        self.assertEqual(
            get_knapsack([], 4 * MiB, lambda n: 0, lambda n: 0.0, "at_least"), []
        )


class TestOptimizerStateBytes(unittest.TestCase):
    def test_no_optimizer_is_zero(self):
        self.assertEqual(optimizer_state_bytes(None, 1000), 0)

    def test_adamw_is_two_fp32_states_per_param(self):
        self.assertEqual(optimizer_state_bytes(_FakeOptConfig(), 1000), 1000 * 2 * 4)

    def test_bf16_fused_states_halve_the_size(self):
        cfg = _FakeOptConfig(implementation="fused_opt_states_bf16")
        self.assertEqual(optimizer_state_bytes(cfg, 1000), 1000 * 2 * 2)

    def test_amsgrad_adds_a_state(self):
        cfg = _FakeOptConfig(param_groups=[_FakeParamGroup("Adam", {"amsgrad": True})])
        self.assertEqual(optimizer_state_bytes(cfg, 100), 100 * 3 * 4)

    def test_sgd_is_stateless_until_momentum(self):
        plain = _FakeOptConfig(param_groups=[_FakeParamGroup("SGD", {})])
        self.assertEqual(optimizer_state_bytes(plain, 100), 0)
        momentum = _FakeOptConfig(
            param_groups=[_FakeParamGroup("SGD", {"momentum": 0.9})]
        )
        self.assertEqual(optimizer_state_bytes(momentum, 100), 100 * 1 * 4)

    def test_unknown_optimizer_reports_the_sentinel(self):
        """Must not fall back to a guess: the caller subtracts this from a
        budget, so a wrong small number silently overshoots."""
        cfg = _FakeOptConfig(param_groups=[_FakeParamGroup("Shampoo", {})])
        self.assertEqual(optimizer_state_bytes(cfg, 1000), UNKNOWN_OPTIMIZER_BYTES)

    def test_one_unknown_group_poisons_the_whole_estimate(self):
        cfg = _FakeOptConfig(
            param_groups=[_FakeParamGroup("AdamW"), _FakeParamGroup("Shampoo")]
        )
        self.assertEqual(optimizer_state_bytes(cfg, 1000), UNKNOWN_OPTIMIZER_BYTES)

    def test_mixed_known_groups_take_the_max(self):
        cfg = _FakeOptConfig(
            param_groups=[_FakeParamGroup("AdamW"), _FakeParamGroup("Adagrad")]
        )
        self.assertEqual(optimizer_state_bytes(cfg, 10), 10 * 2 * 4)


class TestLocalParamNumel(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertEqual(local_param_numel(None), 0)
        self.assertEqual(local_param_numel([]), 0)

    def test_sums_every_pipeline_stage(self):
        """Counting only model_parts[0] under-reserves optimizer state by
        roughly the stage count when PP > 1."""
        a = torch.nn.Linear(4, 8, bias=False)  # 32
        b = torch.nn.Linear(8, 2, bias=False)  # 16
        self.assertEqual(local_param_numel([a]), 32)
        self.assertEqual(local_param_numel([a, b]), 48)


class TestFmtRanges(unittest.TestCase):
    def test_collapses_runs(self):
        self.assertEqual(_fmt_ranges([0, 1, 2, 3, 4, 7, 9, 10, 11]), "0-4,7,9-11")

    def test_singletons_and_empty(self):
        self.assertEqual(_fmt_ranges([5]), "5")
        self.assertEqual(_fmt_ranges([]), "")

    def test_unsorted_input(self):
        self.assertEqual(_fmt_ranges([3, 1, 2]), "1-3")


class TestSizing(unittest.TestCase):
    def test_concrete_bytes_passes_plain_ints_through(self):
        for n in (0, 1, ROUNDING, 1 << 30):
            self.assertEqual(_concrete_bytes(n), n)

    def test_concrete_bytes_refuses_to_guess_an_unresolvable_size(self):
        """Guessing low makes the budget too generous and the run OOMs, so an
        unresolvable symbolic size must raise rather than default."""

        class _Unresolvable:
            node = None

        with self.assertRaises(RuntimeError):
            _concrete_bytes(_Unresolvable())

    def test_get_size_rounds_up_to_the_allocator_granularity(self):
        # 3 fp32 elements = 12 B, which the allocator rounds to one 512 B block.
        self.assertEqual(get_size(torch.empty(3, dtype=torch.float32)), ROUNDING)
        exact = torch.empty(ROUNDING // 4, dtype=torch.float32)
        self.assertEqual(get_size(exact), ROUNDING)

    def test_get_size_measures_storage_not_view(self):
        base = torch.empty(4096, dtype=torch.float32)
        view = base[:16]
        # A view shares its base storage, so keeping it frees nothing extra.
        self.assertEqual(get_size(view), get_size(base))


class TestBackwardPredicates(unittest.TestCase):
    """_is_backward_node is shared by every policy; _touches_backward is the
    input-inclusive form that only the auto_perf_maxing solver wants."""

    def _graph(self):
        """fwd -> bwd -> consumer, where consumer is fed by backward but is
        not itself a backward node."""
        g = torch.fx.Graph()
        fwd = g.placeholder("fwd")
        fwd.meta["autograd_backward"] = False
        bwd = g.call_function(torch.add, (fwd, 1))
        bwd.meta["autograd_backward"] = True
        consumer = g.call_function(torch.mul, (bwd, 2))
        consumer.meta["autograd_backward"] = False
        return fwd, bwd, consumer

    def test_is_backward_node_looks_only_at_the_node(self):
        fwd, bwd, consumer = self._graph()
        self.assertFalse(_is_backward_node(fwd))
        self.assertTrue(_is_backward_node(bwd))
        # The distinguishing case: fed by backward, but not itself backward.
        self.assertFalse(_is_backward_node(consumer))

    def test_touches_backward_includes_inputs(self):
        fwd, bwd, consumer = self._graph()
        self.assertFalse(_touches_backward(fwd))
        self.assertTrue(_touches_backward(bwd))
        self.assertTrue(_touches_backward(consumer))


class TestFsdpShardDegree(unittest.TestCase):
    def _graph_with_all_gather(self, *group_sizes):
        g = torch.fx.Graph()
        x = g.placeholder("x")
        outs = [
            g.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                (x, gs, "0"),
            )
            for gs in group_sizes
        ]
        g.output(tuple(outs) if outs else (x,))
        return torch.fx.GraphModule(torch.nn.Module(), g)

    def test_reads_group_size_off_the_collective(self):
        """LOCAL_WORLD_SIZE equals the shard degree only when the shard group is
        exactly one host, so multi-node runs must read the graph instead."""
        self.assertEqual(_fsdp_shard_degree(self._graph_with_all_gather(16)), 16)

    def test_takes_the_largest_group(self):
        self.assertEqual(_fsdp_shard_degree(self._graph_with_all_gather(2, 16)), 16)

    def test_falls_back_when_there_is_no_collective(self):
        self.assertGreaterEqual(_fsdp_shard_degree(self._graph_with_all_gather()), 1)


class TestResolveMemoryBudget(unittest.TestCase):
    """-1 infers the budget from the local device, so selecting the policy is
    enough to use it."""

    def test_explicit_budget_wins(self):
        cfg = AutoSacConfig(memory_budget_gb=76.0)
        self.assertEqual(resolve_memory_budget_gib(cfg), 76.0)

    def test_large_explicit_budget_is_not_read_as_unset(self):
        """The sentinel is negative precisely so this is honoured."""
        cfg = AutoSacConfig(memory_budget_gb=1200.0)
        self.assertEqual(resolve_memory_budget_gib(cfg), 1200.0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a device to infer from")
    def test_unset_infers_a_fraction_of_device_memory(self):
        total = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory / (1 << 30)
        for frac in (0.90, 0.50):
            cfg = AutoSacConfig(memory_budget_gb=-1.0, budget_fraction=frac)
            self.assertAlmostEqual(
                resolve_memory_budget_gib(cfg), frac * total, places=4
            )

    def test_unset_without_a_device_reports_no_budget(self):
        """0.0 tells the caller to leave the graph untagged rather than
        planning against a number it invented."""
        cfg = AutoSacConfig(memory_budget_gb=-1.0)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(resolve_memory_budget_gib(cfg), 0.0)


class TestValidateFractions(unittest.TestCase):
    def test_accepts_a_normalised_split(self):
        _validate_fractions(0.5, 0.3, 0.2)

    def test_rejects_a_split_that_does_not_sum_to_one(self):
        with self.assertRaises(ValueError):
            _validate_fractions(0.5, 0.3, 0.9)

    def test_rejects_negative_fractions(self):
        with self.assertRaises(ValueError):
            _validate_fractions(-0.1, 0.6, 0.5)


if __name__ == "__main__":
    unittest.main()
