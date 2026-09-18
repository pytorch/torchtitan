# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the two-level ILP memory policy (``auto_perf_maxing``).

Layers:
  * ``TestResolveFractions`` -- fast, CPU-only pure-function checks of the
    keep/recompute/offload split validation.
  * ``TestTwoLevelILP`` -- GPU integration on the llama3 debug model:
      - numerics are preserved across full-keep / sac_and_offload / ILP,
      - a tighter budget makes the ILP free more (materialize more
        recompute/offload ops), and
      - an impossibly small budget makes the outer LP infeasible and no-op.

    NOTE on peaks: on the tiny debug model the recompute working set + offload
    staging buffers exceed the activation bytes freed, so the measured CUDA peak
    does NOT drop under a memory policy (full-keep is actually the lowest). Real
    peak reduction is a large-model property and is validated by the multi-GPU
    integration tests, not here. These tests therefore assert on numerics and on
    the ILP's graph-level response (how much it frees), which is monotonic at any
    scale.

The policy depends on the external ``torchinsights`` package for its estimators,
so the whole module is skipped when torchinsights is unavailable (e.g. CI that
does not install it).
"""

import copy
import os
import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    import torchinsights  # noqa: F401

    _HAS_TORCHINSIGHTS = True
except ImportError:
    _HAS_TORCHINSIGHTS = False

DTYPE = torch.bfloat16
BATCH_SIZE = 2
SEQ_LEN = 2048
DEBUGMODEL = "debugmodel"

# Absolute per-rank budgets (GiB) for the llama3 debug model at the shape above.
# Both are below the ~0.42 GiB all-keep active peak so the ILP engages, and both
# are feasible (above the param/grad floor). Derived from measurement; keep in
# sync if the debug-model shape changes.
LOOSE_BUDGET_GB = 0.35
TIGHT_BUDGET_GB = 0.25
INFEASIBLE_BUDGET_GB = 0.001


@unittest.skipUnless(_HAS_TORCHINSIGHTS, "requires torchinsights")
class TestResolveFractions(unittest.TestCase):
    """Pure-function checks for ``_resolve_fractions`` (no trace/GPU/optimizer)."""

    _ENV_KEYS = ("AUTOAC_KEEP_FRAC", "AUTOAC_RECOMPUTE_FRAC", "AUTOAC_OFFLOAD_FRAC")

    def setUp(self):
        # _resolve_fractions consults AUTOAC_* env overrides; snapshot and clear
        # so tests are isolated, then restore in tearDown.
        self._saved_env = {k: os.environ.pop(k, None) for k in self._ENV_KEYS}

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_valid_split_passes_through(self):
        from torchtitan.experiments.graph_trainer.two_level_ilp_memory_policy_pass import (  # noqa: E501
            _resolve_fractions,
        )

        self.assertEqual(_resolve_fractions(0.05, 0.90, 0.05), (0.05, 0.90, 0.05))
        self.assertEqual(_resolve_fractions(1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_split_not_summing_to_one_raises(self):
        from torchtitan.experiments.graph_trainer.two_level_ilp_memory_policy_pass import (  # noqa: E501
            _resolve_fractions,
        )

        with self.assertRaises(ValueError):
            _resolve_fractions(0.5, 0.4, 0.2)  # sums to 1.1

    def test_negative_fraction_raises(self):
        from torchtitan.experiments.graph_trainer.two_level_ilp_memory_policy_pass import (  # noqa: E501
            _resolve_fractions,
        )

        with self.assertRaises(ValueError):
            _resolve_fractions(-0.1, 1.0, 0.1)

    def test_env_overrides_take_precedence(self):
        from torchtitan.experiments.graph_trainer.two_level_ilp_memory_policy_pass import (  # noqa: E501
            _resolve_fractions,
        )

        os.environ["AUTOAC_KEEP_FRAC"] = "0.10"
        os.environ["AUTOAC_RECOMPUTE_FRAC"] = "0.80"
        os.environ["AUTOAC_OFFLOAD_FRAC"] = "0.10"
        # Passed-in values are ignored in favor of the env overrides.
        self.assertEqual(_resolve_fractions(0.5, 0.5, 0.0), (0.10, 0.80, 0.10))


def _set_deterministic() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)


def _build_model(model_flavor: str = DEBUGMODEL, attn_backend: str = "flex") -> nn.Module:
    from torchtitan.experiments.graph_trainer.llama3 import (
        model_registry as llama3_registry,
    )

    model_spec = llama3_registry(model_flavor, attn_backend=attn_backend)
    with torch.device("meta"):
        model = model_spec.model.build()
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=None)
    model.to(dtype=DTYPE)
    model.train()
    return model


def _build_trainer(
    state_dict: dict,
    memory_policy: str,
    *,
    budget_gb: float = 1e6,
    cpu_offload_budget_gb: float = 100.0,
    runtime_est_mode: str = "cost_model",
):
    """Single-GPU GraphTrainer for a given memory policy.

    Loads ``state_dict`` so every trainer sees identical weights, and applies
    ``annotate_module_fqns`` (normally done by the parallelize path) so the ILP's
    per-layer grouping can find the transformer blocks -- without it the outer
    solve reports "no layers found" and no-ops.
    """
    from torchtitan.experiments.graph_trainer.common_utils import (
        annotate_module_fqns,
    )
    from torchtitan.experiments.graph_trainer.llama3 import (
        model_registry as llama3_registry,
    )
    from torchtitan.experiments.graph_trainer.tests._trainer_test_utils import (
        build_minimal_trainer,
    )
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    model = _build_model()
    model.load_state_dict(copy.deepcopy(state_dict))
    annotate_module_fqns(model)

    trainer = build_minimal_trainer(
        model,
        llama3_registry(DEBUGMODEL).model,
        GraphTrainer,
        activation_checkpoint_mode="none",
    )
    trainer.config.compile.memory_policy = memory_policy
    trainer.config.compile.memory_budget_gb = budget_gb
    trainer.config.compile.runtime_est_mode = runtime_est_mode
    trainer.config.compile.cpu_offload_budget_gb = cpu_offload_budget_gb
    # The ILP pass reads config.optimizer for optimizer-state bytes; None -> 0,
    # which is fine for a fwd/bwd-only test step.
    trainer.config.optimizer = None
    return trainer


@dataclass(frozen=True)
class StepResult:
    loss: torch.Tensor
    grads: list
    active_gib: float  # true peak of live bytes -- the memory signal to trust
    reserved_gib: float
    num_nodes: int


def _step(trainer, tokens, labels):
    model = trainer.model_parts[0]
    model.zero_grad(set_to_none=True)
    global_valid_tokens = torch.tensor(labels.numel(), dtype=torch.float, device="cuda")
    positions = (
        torch.arange(tokens.shape[1], device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .expand(tokens.shape[0], tokens.shape[1])
    )
    return trainer.forward_backward_step(
        input_dict={"input": tokens, "positions": positions},
        labels=labels,
        global_valid_tokens=global_valid_tokens,
    )


def _run_step(trainer, tokens, labels) -> StepResult:
    # Warmup: the first step traces/compiles, whose one-time scratch would skew
    # the measured peak; the tagged graph is cached on trainer._traced_step.
    _step(trainer, tokens, labels)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    loss = _step(trainer, tokens, labels)
    torch.cuda.synchronize()
    model = trainer.model_parts[0]
    grads = [p.grad.detach().clone() for p in model.parameters()]
    stats = torch.cuda.memory_stats()
    return StepResult(
        loss=loss.detach().clone(),
        grads=grads,
        active_gib=stats["active_bytes.all.peak"] / 1e9,
        reserved_gib=torch.cuda.max_memory_reserved() / 1e9,
        num_nodes=len(trainer._traced_step.gm.graph.nodes),
    )


@unittest.skipUnless(
    _HAS_TORCHINSIGHTS and torch.cuda.is_available(), "requires torchinsights + CUDA"
)
class TestTwoLevelILP(unittest.TestCase):
    def setUp(self):
        _set_deterministic()
        # One shared set of weights so runs differ only in memory policy/budget.
        model = _build_model()
        self.state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        del model
        torch.cuda.empty_cache()
        self.tokens = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")
        self.labels = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _run(self, memory_policy, **kwargs) -> StepResult:
        trainer = _build_trainer(self.state_dict, memory_policy, **kwargs)
        result = _run_step(trainer, self.tokens, self.labels)
        del trainer
        torch.cuda.empty_cache()
        return result

    def _assert_same_numerics(self, ref: StepResult, other: StepResult, tag: str):
        self.assertTrue(
            torch.equal(ref.loss, other.loss),
            f"{tag}: loss mismatch ref={ref.loss.item()} other={other.loss.item()}",
        )
        for i, (g_ref, g_other) in enumerate(
            zip(ref.grads, other.grads, strict=True)
        ):
            self.assertTrue(torch.equal(g_ref, g_other), f"{tag}: grad[{i}] mismatch")

    def test_ilp_targets_reference_peaks_and_falls_back_when_infeasible(self):
        """Narrative test:

        1. Measure reference active peaks (full-keep, default SAC, sac_and_offload)
           -- logged for visibility.
        2. Aim the ILP at the default and sac_and_offload peaks (pass each as the
           budget), under both runtime-cost modes (cost_model and benchmark), and
           check it ENGAGES toward the target (materializes recompute/offload ops)
           with unchanged numerics.
        3. Starve the ILP with a tiny budget: the outer LP is infeasible and the
           policy falls back to the full-keep graph (no-op), numerics unchanged.

        IMPORTANT -- why real peak is NOT asserted here: in one process the
        measured active peak drifts ~18% by measurement position (compiled
        artifacts / allocator residue accumulate across trainers), which is larger
        than the ~15% gap between policies. So peak-*matching* is validated by the
        multi-process integration tests (one torchrun per config, no drift); this
        unit test asserts only the drift-immune signals: numerics, graph
        engagement/monotonicity, and infeasible fallback.
        """
        # --- 1. reference peaks (logged, not asserted; see docstring) ---
        full_keep = self._run("auto_perf_maxing", budget_gb=1e6)  # no-op = all keep
        default = self._run("default")
        sac_off = self._run("sac_and_offload")
        print(
            f"[two_level_ilp] ref active peak GiB: full_keep={full_keep.active_gib:.4f} "
            f"default={default.active_gib:.4f} sac_and_offload={sac_off.active_gib:.4f}"
        )

        # --- 2. aim the ILP at each reference peak, under both runtime modes ---
        for mode in ("cost_model", "benchmark"):
            ilp_default = self._run(
                "auto_perf_maxing", budget_gb=default.active_gib, runtime_est_mode=mode
            )
            ilp_sac = self._run(
                "auto_perf_maxing", budget_gb=sac_off.active_gib, runtime_est_mode=mode
            )
            print(
                f"[two_level_ilp/{mode}] ilp@default="
                f"{ilp_default.active_gib:.4f}GiB/{ilp_default.num_nodes}n "
                f"ilp@sac={ilp_sac.active_gib:.4f}GiB/{ilp_sac.num_nodes}n"
            )
            # Engagement: the ILP frees ops the no-op full-keep graph lacks.
            self.assertGreater(
                ilp_default.num_nodes,
                full_keep.num_nodes,
                f"{mode}: ILP@default did not engage",
            )
            self.assertGreater(
                ilp_sac.num_nodes,
                full_keep.num_nodes,
                f"{mode}: ILP@sac did not engage",
            )
            # Tighter target (sac < default) frees at least as much.
            self.assertGreaterEqual(
                ilp_sac.num_nodes,
                ilp_default.num_nodes,
                f"{mode}: tighter target did not free >= looser "
                f"(sac={ilp_sac.num_nodes} default={ilp_default.num_nodes})",
            )
            # Numerics-preserving under both runtime modes.
            self._assert_same_numerics(full_keep, ilp_default, f"{mode}:ilp@default")
            self._assert_same_numerics(full_keep, ilp_sac, f"{mode}:ilp@sac")

        # --- 3. infeasible budget -> full-keep fallback ---
        from torchtitan.tools import logging as tt_logging

        with self.assertLogs(tt_logging.logger, level="WARNING") as cm:
            infeasible = self._run(
                "auto_perf_maxing", budget_gb=INFEASIBLE_BUDGET_GB
            )
        self.assertTrue(
            any(
                "not optimal" in m.lower() or "infeasible" in m.lower()
                for m in cm.output
            ),
            f"expected an infeasibility warning, got: {cm.output}",
        )
        # Fallback == full-keep graph, and numerics unchanged.
        self.assertEqual(infeasible.num_nodes, full_keep.num_nodes)
        self._assert_same_numerics(full_keep, infeasible, "infeasible-fallback")

    def test_tighter_budget_frees_more(self):
        """The ILP's knob-response: a lower budget must materialize strictly more
        recompute/offload ops than a looser budget (and than full-keep). This is
        the deterministic, scale-independent form of "budget controls the peak".
        """
        full_keep = self._run("auto_perf_maxing", budget_gb=1e6)
        loose = self._run("auto_perf_maxing", budget_gb=LOOSE_BUDGET_GB)
        tight = self._run("auto_perf_maxing", budget_gb=TIGHT_BUDGET_GB)

        self.assertGreater(
            loose.num_nodes,
            full_keep.num_nodes,
            f"ILP did not engage: loose={loose.num_nodes} "
            f"full_keep={full_keep.num_nodes}",
        )
        self.assertGreater(
            tight.num_nodes,
            loose.num_nodes,
            "tighter budget did not free more: "
            f"tight={tight.num_nodes} loose={loose.num_nodes}",
        )


if __name__ == "__main__":
    unittest.main()
