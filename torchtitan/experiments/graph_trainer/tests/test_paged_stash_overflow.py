# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the non-blocking paged-stash overflow check.

Single GPU, no model: the check state is built around a bare overflow flag
via ``_init_overflow_check_state``, which is exactly the surface the deferred
check operates on.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import PagedStash


def _make_paged_stash(device: str = "cuda") -> PagedStash:
    """Build a PagedStash with only the overflow-check surface populated."""
    ps = PagedStash.__new__(PagedStash)
    ps.buffers = {}
    ps.host_spill = None
    ps.overflow = torch.zeros(1, dtype=torch.int64, device=device)
    ps._init_overflow_check_state()
    return ps


class TestPagedStashOverflowCheck(TestCase):
    def test_no_overflow_never_raises(self):
        ps = _make_paged_stash()
        for _ in range(4):
            ps.check_overflow()
            torch.cuda.synchronize()
        ps.check_overflow_blocking()

    def test_deferred_check_raises_within_two_calls(self):
        ps = _make_paged_stash()
        ps.check_overflow()  # enqueue first mirror copy (flag still 0)
        ps.overflow.fill_(1)  # device-side overflow, as the copy kernel would set it
        ps.check_overflow()  # samples the flag; not yet observed host-side
        torch.cuda.synchronize()  # let the async copy land
        with self.assertRaisesRegex(RuntimeError, "Paged stash buffer overflow"):
            ps.check_overflow()

    def test_flag_is_sticky_across_reset(self):
        ps = _make_paged_stash()
        ps.overflow.fill_(1)
        ps.reset()  # must NOT clear the overflow flag
        ps.check_overflow()  # enqueue mirror copy of the (sticky) flag
        torch.cuda.synchronize()
        with self.assertRaisesRegex(RuntimeError, "Paged stash buffer overflow"):
            ps.check_overflow()

    def test_blocking_check_catches_last_step_overflow(self):
        # An overflow after the final pre-step hook is only observable by the
        # shutdown check: no later check_overflow() call exists to read it.
        ps = _make_paged_stash()
        ps.check_overflow()  # last hook of the run (flag still 0)
        ps.overflow.fill_(1)  # overflow during the final step
        with self.assertRaisesRegex(RuntimeError, "Paged stash buffer overflow"):
            ps.check_overflow_blocking()

    def test_check_is_nonblocking(self):
        ps = _make_paged_stash()
        # With ~1s of work in flight on the stream, the deferred check must
        # return without draining it.
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(2_000_000_000)  # ~1s at ~2GHz
        ps.check_overflow()
        end.record()
        still_running = not end.query()  # sleep still on the stream => no sync
        torch.cuda.synchronize()
        self.assertTrue(still_running)


if __name__ == "__main__":
    run_tests()
