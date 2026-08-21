# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the vision encodes inside the schedule's idle time, on the main stream.

    The companion to :mod:`dep_bubble_plan`, which decides WHERE each encode goes; this
    puts it there. The hook fires AFTER a forward action returns rather than before a
    receive wait, because the rank owning the tower owns stage 0, whose forward receives
    nothing.

    See ``phase13_k3like_48b_posttrain/DEP_BUBBLE_RUNTIME.md``.
    """

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from torchtitan.models.kimi_k3.dep_bubble_plan import BubblePlan

from torchtitan.tools.logging import logger


class _AnchorFirer:
    """Runs the planned encodes after the action they are anchored to.

    Anchored on the action BEFORE the idle interval and fired after it returns, so the
    encode occupies the gap from its start. The first attempt anchored on the action
    AFTER the interval and hooked ``fwd_recv_ops.pop`` -- the moment the runtime is
    about to wait for a receive. Correct in principle and useless in practice: the rank
    owning the tower owns pipeline stage 0, whose forward receives nothing, so no pop
    ever happens for it. That version planned 8 placements on a real pp8xvp4 cell and
    fired 0, which the fired-vs-placed warning reported rather than hiding.
    """

    def __init__(self, on_anchor) -> None:
        self._on_anchor = on_anchor
        self._by_anchor: dict[tuple[int, int], list[int]] = {}
        self.fired = 0
        # Wall-clock actually spent inside the planned encodes, and the count of them.
        # The plan is built from a STATIC cost ratio, and this session paid for the gap
        # that leaves: a ratio measured at seq 4096 (0.493) was handed to a seq-256 cell
        # where the true value is about 14, so each encode overran its interval roughly
        # 28-fold. Every counter still read green, because "ran at the planned point" was
        # true -- occupancy is not hiding. Measuring the encodes is what makes that
        # visible as something other than a slower step.
        self.encode_seconds = 0.0
        self.encode_calls = 0

    def arm(self, plan: BubblePlan) -> None:
        """Load this step's placements. Called once per step, before the loop."""
        self._by_anchor = {}
        for placement in plan.placed:
            kind, stage_index, mb_index = placement.anchor
            if "FORWARD" not in kind:
                # Backward anchors need the adapter's gradient path; forward first.
                continue
            self._by_anchor.setdefault((stage_index, mb_index), []).append(
                placement.microbatch
            )

    def after_forward(self, stage_index: int, mb_index: int) -> None:
        queued = self._by_anchor.pop((stage_index, mb_index), None)
        if not queued:
            return
        # perf_counter around a CUDA call measures launch, not execution, unless the
        # stream is synchronized. The encodes run on the MAIN stream and the next
        # pipeline action is issued to it immediately, so a sync here would serialize
        # what the mechanism exists to overlap. Timing the launch window is still worth
        # having: an encode whose kernels do not fit the interval shows up as the launch
        # blocking on a full queue, and the step-time comparison remains the real
        # measurement.
        start = time.perf_counter()
        self._on_anchor(queued)
        self.encode_seconds += time.perf_counter() - start
        self.encode_calls += len(queued)
        self.fired += len(queued)


def _wrap_stage_forwards(pp_schedule, firer: _AnchorFirer) -> int:
    """Call ``firer.after_forward`` after each stage's ``forward_one_chunk`` returns.

    Wrapped outermost and marked, so the adapter's own micro-batch-index patch on the
    same method keeps working -- that one runs first and unconditionally under DEP, and
    double-wrapping it was already a known way to break it.
    """
    wrapped = 0
    for stage in getattr(pp_schedule, "_stages", []) or []:
        if getattr(stage, "_kimi_bubble_wrapped", False):
            continue
        inner = stage.forward_one_chunk
        stage_index = int(getattr(stage, "stage_index", -1))

        def make(inner=inner, stage_index=stage_index):
            def forward_one_chunk(fwd_chunk_id, *args, **kwargs):
                out = inner(fwd_chunk_id, *args, **kwargs)
                firer.after_forward(stage_index, int(fwd_chunk_id))
                return out

            return forward_one_chunk

        stage.forward_one_chunk = make()  # type: ignore[method-assign]
        stage._kimi_bubble_wrapped = True
        wrapped += 1
    return wrapped


def install_bubble_runtime(
    pp_schedule,
    *,
    plan_for_step: Callable[[], BubblePlan | None],
    encode_now: Callable[[Sequence[int]], None],
    upfront_encode: Callable[[Sequence[int]], None],
) -> None:
    """Make ``pp_schedule`` run planned vision encodes in its idle intervals.

    ``plan_for_step`` returns this rank's plan, or None to leave the schedule alone --
    which is how a step with no visual items, or a rank owning no vision work, opts out
    without a second code path.

    ``encode_now`` runs the encodes on the current (main) stream. ``upfront_encode``
    runs the report's synchronous prefix before the action loop.

    Patches the instance, not the class: torchtitan chooses which schedule class to
    build, and the same reasoning already applies to the cross-stage adapter's own
    ``step`` patch next door.
    """
    if getattr(pp_schedule, "_kimi_bubble_runtime", False):
        return
    firer = _AnchorFirer(encode_now)
    if not _wrap_stage_forwards(pp_schedule, firer):
        raise AttributeError(
            "no pipeline stages on this schedule to wrap: the bubble runtime fires "
            "after a stage's forward_one_chunk, so a schedule without _stages cannot "
            "host it."
        )
    orig_step = pp_schedule.step

    def patched_step(*args, **kwargs):
        plan = plan_for_step()
        if plan is None:
            return orig_step(*args, **kwargs)
        firer.arm(plan)
        before = firer.fired
        if plan.upfront:
            # The report's own design: the first micro-batches' encodes cannot be
            # placed, because nothing precedes them.
            upfront_encode(plan.upfront)
        try:
            return orig_step(*args, **kwargs)
        finally:
            placed = len(plan.placed)
            fired = firer.fired - before
            # Placed-but-never-fired means the anchor action did not run on this rank
            # this step, i.e. the plan and the schedule disagree. Silence there would
            # let the encode fall back to its synchronous path and still look correct.
            level = logger.info if fired == placed else logger.warning
            # Encode time per call alongside the counts, because the counts alone
            # cannot distinguish "hidden in the bubble" from "ran at the planned point
            # and overran it". The budget comes from a static cost ratio, so a ratio taken
            # at another sequence length makes every placement look green while overrunning.
            per = (
                firer.encode_seconds / firer.encode_calls if firer.encode_calls else 0.0
            )
            # idle_slots is the one number that separates "the schedule has no bubbles"
            # from "it has bubbles and the planner under-placed". The planner places at
            # most ONE encode per idle slot, so placed can never exceed it however small
            # the cost ratio gets -- which is exactly the case dynamic CP creates, since
            # it divides the per-rank encoder cost before DEP ever sees it. Without this
            # printed, a run showing 4 placements out of 64 looks the same either way.
            level(
                "DEP bubble runtime: %d/%d planned encode(s) ran in a bubble, "
                "%d upfront, %d left synchronous, %d idle slot(s) "
                "(%d starved, %d exhausted), %.1f ms per planned encode",
                fired,
                placed,
                len(plan.upfront),
                len(plan.synchronous),
                plan.idle_slots,
                plan.slots_starved,
                plan.slots_exhausted,
                per * 1e3,
            )
            firer.encode_seconds = 0.0
            firer.encode_calls = 0

    pp_schedule.step = patched_step  # type: ignore[method-assign]
    pp_schedule._kimi_bubble_runtime = True
