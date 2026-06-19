# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Benchmark instrumentation mixin for the FlexShard Muon containers."""

from __future__ import annotations

import os
import time

import torch

from torchtitan.tools.logging import logger

from . import comm_counter


class _BenchMixin:
    """Opt-in systems instrumentation for the distributed-Muon benchmark.

    Enabled by ``FLEX_SHARD_MUON_BENCH=1`` (otherwise ``step`` is the base method,
    zero overhead). When on, each step reports the whole-lifecycle breakdown:

    * ``total_iter`` -- wall-clock per iteration, measured directly as the time
      between consecutive ``step()`` calls (the same ``perf_counter`` method the
      trainer uses for ``time_metrics/end_to_end``); not derived from tps.
    * ``opt_step`` -- isolated ``optimizer.step()`` GPU time (CUDA events).
    * ``fwd_bwd`` -- ``total_iter - opt_step`` (forward + backward + overhead).
    * ``step_comm`` / ``fwdbwd_comm`` -- comm bytes inside the step (NS all-gather
      for the gather baselines; ~0 for ``Owned``) vs. during fwd/bwd (param unshard
      + gradient reduce), from the comm-byte counter snapshot around the step.

    ``opt_step``, ``step_comm`` and ``fwdbwd_comm`` are batch-independent (params /
    matrices / gradients); ``total_iter``, ``fwd_bwd`` and peak memory scale with
    batch. CUDA-event reads use the *previous* step (one-iteration lag) so they never
    block the current iteration; the last step is unreported (negligible over a
    >=10-step run). Peak memory and tps/mfu come from the trainer's metrics logger.

    MuonClip caveat: when ``qk_clip_tau`` is set, QK-clip adds an extra per-forward
    ``[B, H, S, S]`` attention-logit capture pass (see :func:`qkclip._max_logit_per_head`)
    that lands in ``fwd_bwd``, and a small per-head ``all_reduce(MAX)`` plus the row
    rescale in ``opt_step`` -> ``step_comm`` (so the step is collective-free only
    *without* QK-clip). Both scale with QK-clip being on, not with the dense-Muon
    strategy, so to isolate QKClip's cost compare a config against the same config with
    ``qk_clip_tau=None`` rather than against AdamW.
    """

    def _bench_setup(self) -> None:
        self._bench_on = os.environ.get("FLEX_SHARD_MUON_BENCH") == "1"
        if not self._bench_on:
            return
        comm_counter.install()
        comm_counter.reset()
        self._bench_idx = 0
        self._bench_warmup = int(os.environ.get("FLEX_SHARD_MUON_BENCH_WARMUP", "3"))
        self._bench_last_comm = comm_counter.read()
        self._bench_pending = None
        self._bench_n = 0
        self._bench_sum_iter_ms = 0.0
        self._bench_sum_opt_ms = 0.0
        self._bench_sum_step_mb = 0.0
        self._bench_sum_fb_mb = 0.0

    def _bench_flush(self, iter_end: float) -> None:
        if self._bench_pending is None:
            return
        idx, entry_time, start, end, step_bytes, fwdbwd_bytes = self._bench_pending
        self._bench_pending = None
        if idx < self._bench_warmup:
            return
        # total_iter is measured directly (wall-clock between consecutive step()
        # entries); opt_step is the GPU time of this step; fwd_bwd is the remainder.
        total_iter_ms = (iter_end - entry_time) * 1e3
        end.synchronize()  # previous step's event; already complete, ~0 wait
        opt_ms = start.elapsed_time(end)
        fwd_bwd_ms = total_iter_ms - opt_ms
        step_mb = step_bytes / 1e6
        fb_mb = fwdbwd_bytes / 1e6
        self._bench_n += 1
        self._bench_sum_iter_ms += total_iter_ms
        self._bench_sum_opt_ms += opt_ms
        self._bench_sum_step_mb += step_mb
        self._bench_sum_fb_mb += fb_mb
        n = self._bench_n
        logger.info(
            f"[muon-bench] step {idx}: total_iter={total_iter_ms:.3f} ms  "
            f"opt_step={opt_ms:.3f} ms  fwd_bwd={fwd_bwd_ms:.3f} ms  "
            f"step_comm={step_mb:.2f} MB  fwdbwd_comm={fb_mb:.2f} MB  | "
            f"avg/{n}: total_iter={self._bench_sum_iter_ms / n:.3f} ms  "
            f"opt_step={self._bench_sum_opt_ms / n:.3f} ms  "
            f"fwd_bwd={(self._bench_sum_iter_ms - self._bench_sum_opt_ms) / n:.3f} ms  "
            f"step_comm={self._bench_sum_step_mb / n:.2f} MB  "
            f"fwdbwd_comm={self._bench_sum_fb_mb / n:.2f} MB"
        )

    def _post_step(self) -> None:
        """Hook run after the wrapped optimizer step (e.g. MuonClip QK-clip).

        No-op by default; Muon containers override it to apply :class:`QKClip`. Runs
        inside the benched ``opt_step`` window so its cost is attributed to the step.
        """

    def step(self, closure=None):
        if not hasattr(self, "_bench_on"):
            self._bench_setup()
        if not self._bench_on:
            out = super().step(closure)
            self._post_step()
            return out
        # Mark iteration boundary and flush the previous step (its iteration is now
        # complete, and its CUDA events are done -> no live sync).
        now = time.perf_counter()
        self._bench_flush(now)
        comm_at_entry = comm_counter.read()
        fwdbwd_bytes = comm_at_entry - self._bench_last_comm
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = super().step(closure)
        self._post_step()
        end.record()
        comm_at_exit = comm_counter.read()
        self._bench_last_comm = comm_at_exit
        self._bench_pending = (
            self._bench_idx,
            now,
            start,
            end,
            comm_at_exit - comm_at_entry,
            fwdbwd_bytes,
        )
        self._bench_idx += 1
        return out
