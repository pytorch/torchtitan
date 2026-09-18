# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Budget adherence for ``auto_perf_maxing`` on llama3-8B under FSDP.

The solver's other tests cover pure helpers. This covers the property the
policy exists to provide: ask for a peak, get at most that peak. It is only
observable once a plan has been materialized and executed, which is why this
runs the real pass pipeline and real steps rather than inspecting the solver.

Two phases, mirroring how the policy is used:

1. ``full`` recompute establishes the floor -- the lowest peak any policy can
   reach, and a plan the solver can always satisfy.
2. ``auto_perf_maxing`` is budgeted just above that floor, the tightest
   request that is provably feasible, and the measured peak must honour it.

Runs in the H100 suite. Per-rank memory depends on the shard degree -- at
dp_shard=8 the persistent state is ~8 GiB and the measured peak lands in the
teens -- but the gate keeps it off the 24 GiB A10G runners of the default
suite, where headroom over the CUDA context and NCCL buffers is thin. There,
the cover for this policy is the ``auto_perf_maxing`` entry in
integration_tests.py, which checks it runs rather than what it peaks at.
"""

from __future__ import annotations

import logging
import unittest

import torch
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.tests._graph_step_harness import (
    build_compile_config,
    GraphStepHarness,
)

logger = logging.getLogger(__name__)

MODEL_FLAVOR = "8B"
SEQ_LEN = 512
NUM_TOKENS = 8 * SEQ_LEN
VOCAB = 2048

# Budget handed to phase 2, above the measured full-recompute floor.
BUDGET_HEADROOM_GIB = 1.0
# Allowance on top of it. Absolute, not a fraction: 10% of a ~64 GiB peak would
# be 6 GiB, far wider than the headroom under test, which would make the
# assertion vacuous. This covers allocator rounding and NCCL buffers.
OVERSHOOT_TOLERANCE_GIB = 1.5

# Gate so this skips on the 24 GiB A10G runners of the default suite and runs
# on H100s. Deliberately below 79: an 80 GB H100 reports ~79.2 GiB, so a gate
# of 80 would skip on the very hardware this targets.
MIN_DEVICE_MEMORY_GIB = 40


def _device_memory_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1 << 30)


class TestAutoPerfMaxingBudget(FSDPTest):
    """FSDP-only. TP and PP are out of scope.

    ``_fsdp_shard_degree`` takes the largest all_gather group size in the
    graph, which only coincides with the FSDP group when no other collective
    group is present. Adding TP would exercise that ambiguity rather than
    budget adherence.
    """

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def _harness(self) -> GraphStepHarness:
        parallel_dims = ParallelDims(
            dp_shard=self.world_size,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        return GraphStepHarness(
            model_flavor=MODEL_FLAVOR,
            parallel_dims=parallel_dims,
            loss_fn=CrossEntropyLoss.Config().build(),
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >= 2 GPUs")
    @unittest.skipUnless(
        _device_memory_gib() >= MIN_DEVICE_MEMORY_GIB,
        f"llama3-{MODEL_FLAVOR} needs >= {MIN_DEVICE_MEMORY_GIB} GiB/rank",
    )
    def test_measured_peak_honours_the_budget(self):
        harness = self._harness()
        call_args = harness.make_inputs(NUM_TOKENS, SEQ_LEN, VOCAB)

        full = harness.trace_and_compile(
            call_args, build_compile_config(memory_policy="full")
        )
        floor = harness.run_and_measure(full, call_args)
        del full

        budget = floor + BUDGET_HEADROOM_GIB
        auto = harness.trace_and_compile(
            call_args,
            build_compile_config(
                memory_policy="auto_perf_maxing", memory_budget_gb=budget
            ),
        )
        measured = harness.run_and_measure(auto, call_args)

        logger.info(
            "auto_perf_maxing: full-recompute floor %.3f GiB, budget %.3f GiB, "
            "measured %.3f GiB",
            floor,
            budget,
            measured,
        )

        self.assertLessEqual(
            measured,
            budget + OVERSHOOT_TOLERANCE_GIB,
            f"auto_perf_maxing exceeded its budget: measured {measured:.3f} GiB "
            f"> {budget:.3f} + {OVERSHOOT_TOLERANCE_GIB:.3f} tolerance "
            f"(full-recompute floor {floor:.3f} GiB)",
        )


if __name__ == "__main__":
    unittest.main()
