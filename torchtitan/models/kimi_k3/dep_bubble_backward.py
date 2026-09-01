# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Defer the vision tower's backward so it can run in a pipeline bubble.

Report sec 5.2.3: "the backward passes are handled analogously". The forward half only
had to decide WHEN to call the encode, because nothing else consumes it. The backward
has no such freedom by default: the tower's output is spliced into the text embedding
and the two share one autograd graph, so the tower's backward happens inside the
splicing stage's ``backward_one_chunk``, inline, wherever the schedule put that action.

Moving it means cutting the graph at the seam. :func:`cut_for_deferred_backward`
detaches the tower's output, splices the detached stand-in, and captures the gradient
that arrives on it with a tensor hook; the tower's own backward is then replayed later,
explicitly, at a planned slot.

## The invariant that matters more than the placement

Every deferred backward MUST run before the optimizer step. A gradient that was cut off
and never re-run is not a slow step, it is silently wrong training: the tower's
parameters simply do not get that micro-batch's contribution, and nothing raises.
:meth:`GradQueue.drain` is therefore called unconditionally at step end for whatever the
plan did not place, and :meth:`GradQueue.assert_empty` exists so a caller can turn a
leak into an exception rather than a quiet accuracy loss.

Ordering does not matter for correctness -- parameter gradients accumulate -- so a
deferred backward is free to run in any bubble after its gradient arrives. What it costs
is memory: the tower's forward graph for that micro-batch has to stay alive from the
encode until the deferred backward runs, which is a longer window than the forward
prefetch's and is the real bound on how much of the backward can be moved.
"""

from __future__ import annotations

import torch

from torchtitan.tools.logging import logger


def cut_for_deferred_backward(
    features: torch.Tensor, queue: "GradQueue", microbatch: int
) -> torch.Tensor:
    """Return a stand-in for ``features`` whose gradient is queued, not propagated.

    Splice the RESULT into the text embedding. The tower's graph stays alive and
    untouched until :meth:`GradQueue.run_one` or :meth:`GradQueue.drain` replays the
    captured gradient into it.

    A detached leaf plus a tensor hook, and both halves of that are load-bearing.

    The detach makes the tower's graph unreachable from the text's, which is what keeps
    the text's ``.backward()`` from freeing it. An ``autograd.Function`` wrapping the
    tower's output does NOT achieve that even when its backward returns ``None``:
    measured, the deferred pass then dies with "Trying to backward through the graph a
    second time", and it only survives if the text backward is given
    ``retain_graph=True`` -- which would mean holding the whole text graph for the sake
    of the tower, and the pipeline calls that backward itself.

    The hook, rather than a Function, is what fires at the right moment without putting
    anything back in the graph: ``detached`` is a leaf of the text graph, so autograd
    computes its gradient and calls the hook there. Returning ``None`` from the hook
    leaves that gradient as it is; returning anything else would rewrite it.
    """
    if not features.requires_grad:
        # Nothing to defer: the tower has no gradient path this step (normal
        # under LoRA with no adapter inside it). Cutting anyway adds a
        # grad-requiring leaf and drags stage_backward down a path it never took.
        return features

    detached = features.detach().requires_grad_(True)

    def _capture(grad: torch.Tensor):
        queue.stash(microbatch, features, grad)
        return None

    detached.register_hook(_capture)
    return detached


class GradQueue:
    """Vision backwards whose gradient has arrived but which have not run yet.

    ``max_pending`` bounds how many may wait at once. Each waiting entry keeps one
    micro-batch's tower forward graph alive from the encode until the replay, which
    is a longer window than the forward prefetch's and is the real limit on how much
    of the backward can be moved. Above the bound the earliest pending entry runs
    immediately, turning the memory window into a configured quantity instead of
    whatever the plan happened to imply.

    Zero means unbounded, and that is the default deliberately. The window has not
    been measured (it needs a box that can hold the configuration where hiding
    exists), so a nonzero default would replace a known behaviour with a guessed
    number. What the bound is for is the run that hits its memory ceiling: there it
    is a knob rather than a rewrite.
    """

    def __init__(self, max_pending: int = 0) -> None:
        self._pending: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._max_pending = max(0, int(max_pending))
        self.ran = 0
        self.drained = 0
        # Ran early because the bound was reached, not because a slot came up.
        self.forced = 0
        # Slots that came up with nothing pending. The greedy placement assumes
        # the earliest micro-batch's gradient arrives first; a high count says
        # that fails for this schedule and min() should become any-pending.
        self.idle_slots = 0

    def stash(self, microbatch: int, output: torch.Tensor, grad: torch.Tensor) -> None:
        self._pending.setdefault(microbatch, []).append((output, grad))
        while self._max_pending and self.pending_count() > self._max_pending:
            before = self.ran
            if not self.run_one(min(self._pending)):
                break
            self.forced += self.ran - before

    def has(self, microbatch: int) -> bool:
        return bool(self._pending.get(microbatch))

    def run_one(self, microbatch: int) -> bool:
        """Run the tower's backward for ``microbatch`` if its gradient has arrived.

        False when it has not: the plan is derived from the schedule's shape, so a slot
        can come up before the gradient does, and that is not an error -- the entry stays
        pending and the step-end drain will take it.
        """
        entries = self._pending.pop(microbatch, None)
        if not entries:
            return False
        for output, grad in entries:
            torch.autograd.backward(output, grad)
            self.ran += 1
        return True

    def run_next(self) -> bool:
        """Run the earliest pending vision backward, if any.

        The backward side is greedy rather than budget-planned, and that is a deliberate
        difference from the forward. A placement plan needs to know when the work becomes
        runnable, and on the forward side that is static -- the pixels are there from step
        entry. A vision backward only becomes runnable once the text backward for its
        micro-batch has produced the gradient, which is a schedule-dependent moment the
        planner would have to model. Taking one pending item per idle interval after a
        backward action is the same placement the plan would make in the common case and
        needs no model of arrival time.
        """
        if not self._pending:
            self.idle_slots += 1
            return False
        return self.run_one(min(self._pending))

    def drain(self) -> int:
        """Run everything still pending. Called unconditionally at step end.

        This is not a fallback for tidiness. A deferred backward that never runs means
        the tower silently misses that micro-batch's gradient, with no error anywhere,
        so the drain is the correctness guarantee and the placement is only the
        optimisation.
        """
        count = 0
        for microbatch in sorted(self._pending):
            for output, grad in self._pending[microbatch]:
                torch.autograd.backward(output, grad)
                count += 1
        self._pending.clear()
        self.drained += count
        return count

    def assert_empty(self, where: str) -> None:
        if self._pending:
            raise AssertionError(
                f"{where}: {sum(len(v) for v in self._pending.values())} vision "
                f"backward(s) still pending for micro-batches "
                f"{sorted(self._pending)}. Running the optimizer now would train the "
                f"tower on incomplete gradients."
            )

    def pending_count(self) -> int:
        return sum(len(v) for v in self._pending.values())

    def report(self, placed: int) -> None:
        level = logger.info if self.drained == 0 else logger.warning
        # forced and idle_slots are the two ways the placement can work against
        # the schedule while every gradient still runs: memory decided the when,
        # or slots found nothing to place. Both are silent in the loss.
        level(
            "DEP bubble backward: %d ran at a planned slot, %d drained at step end, "
            "%d forced by the pending bound, %d slot(s) found nothing pending "
            "(%d slots planned)",
            self.ran - self.forced,
            self.drained,
            self.forced,
            self.idle_slots,
            placed,
        )
        self.ran = 0
        self.drained = 0
        self.forced = 0
        self.idle_slots = 0


def install_backward_slots(pp_schedule, queue: GradQueue) -> int:
    """Run one queued vision backward after each of this rank's backward actions.

    Same shape as the forward's hook and for the same reason: the idle interval starts
    when an action completes, so firing after ``backward_one_chunk`` returns puts the
    work in the gap rather than in front of the next action's wait.

    Also makes ``step`` drain whatever is left. That is not tidiness -- a deferred
    backward that never runs leaves the tower without that micro-batch's gradient and
    raises nothing.
    """
    wrapped = 0
    for stage in getattr(pp_schedule, "_stages", []) or []:
        if getattr(stage, "_kimi_bubble_bwd_wrapped", False):
            continue
        inner = getattr(stage, "backward_one_chunk", None)
        if inner is None:
            continue

        def make(inner=inner):
            def backward_one_chunk(*args, **kwargs):
                out = inner(*args, **kwargs)
                queue.run_next()
                return out

            return backward_one_chunk

        stage.backward_one_chunk = make()  # type: ignore[method-assign]
        stage._kimi_bubble_bwd_wrapped = True
        wrapped += 1

    if not getattr(pp_schedule, "_kimi_bubble_backward_step", False):
        orig_step = pp_schedule.step

        def patched_step(*args, **kwargs):
            try:
                return orig_step(*args, **kwargs)
            finally:
                left = queue.drain()
                queue.report(placed=queue.ran + left)

        pp_schedule.step = patched_step  # type: ignore[method-assign]
        pp_schedule._kimi_bubble_backward_step = True
    return wrapped
