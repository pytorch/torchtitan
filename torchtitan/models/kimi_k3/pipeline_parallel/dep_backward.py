# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hold the vision tower's backward so it can run in a pipeline bubble."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def cut_for_deferred_backward(
    features: torch.Tensor, queue: GradQueue, microbatch: int
) -> torch.Tensor:
    """A stand-in for ``features`` whose gradient is queued instead of propagated."""
    if not features.requires_grad:
        # No gradient path through the tower this step, the normal case under a
        # LoRA that adapts nothing inside it. Cutting anyway would give the
        # splicing stage's output a grad-requiring leaf it did not have.
        return features
    # The detach is what keeps the text backward from freeing the tower's graph;
    # an autograd.Function returning None does not, and the pipeline runs that
    # backward itself, so retain_graph is not ours to pass. The hook fires where
    # the gradient lands without putting anything back in the graph.
    detached = features.detach().requires_grad_(True)

    def _capture(grad: torch.Tensor):
        queue.stash(microbatch, features, grad)
        return None

    detached.register_hook(_capture)
    return detached


class GradQueue:
    """Vision backwards whose gradient has arrived but which have not run yet.

    Each waiting entry keeps one micro-batch's tower graph alive, so
    ``max_pending`` bounds that memory; zero, the default, is unbounded.
    """

    def __init__(self, max_pending: int = 0) -> None:
        self._pending: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._max_pending = max(0, int(max_pending))
        self.ran = 0
        self.drained = 0
        self.forced = 0
        self.idle_slots = 0

    def stash(self, microbatch: int, output: torch.Tensor, grad: torch.Tensor) -> None:
        self._pending.setdefault(microbatch, []).append((output, grad))
        while self._max_pending and self.pending_count() > self._max_pending:
            before = self.ran
            if not self.run_one(min(self._pending)):
                break
            self.forced += self.ran - before

    def run_one(self, microbatch: int) -> bool:
        """Run the tower's backward for ``microbatch``; False if its gradient has not arrived."""
        entries = self._pending.pop(microbatch, None)
        if not entries:
            return False
        for output, grad in entries:
            torch.autograd.backward(output, grad)
            self.ran += 1
        return True

    def run_next(self) -> bool:
        """Run the earliest pending vision backward, if any."""
        if not self._pending:
            self.idle_slots += 1
            return False
        return self.run_one(min(self._pending))

    def drain(self) -> int:
        """Run everything still pending.

        A deferred backward that never runs leaves the tower without that
        micro-batch's gradient and raises nothing, so this is the correctness
        guarantee and the placement is only the optimisation.
        """
        count = 0
        for microbatch in sorted(self._pending):
            for output, grad in self._pending[microbatch]:
                torch.autograd.backward(output, grad)
                count += 1
        self._pending.clear()
        self.drained += count
        return count

    def pending_count(self) -> int:
        return sum(len(v) for v in self._pending.values())

    def report(self) -> None:
        """Log the step's counts and reset them."""
        level = logger.info if self.drained == 0 else logger.warning
        level(
            "DEP bubble backward: %d ran at a planned slot, %d drained at step end, "
            "%d forced by the pending bound, %d slot(s) found nothing pending",
            self.ran - self.forced,
            self.drained,
            self.forced,
            self.idle_slots,
        )
        self.ran = self.drained = self.forced = self.idle_slots = 0
