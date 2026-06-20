# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Warmup-Stable-Decay-Simplified (WSD-S) learning-rate scheduler.

Ported from OLMo-core ``WSDS.get_lr`` (arXiv:2410.05192). A single continuous
sawtooth across the Chinchilla periods: warm up once, then for each period hold
the peak LR and linearly decay to 0; the next period jumps straight back to peak
with no re-warmup. The LR is a pure function of the global step, so the inherited
``state_dict`` (last_epoch only) suffices.
"""

import functools
import itertools
from dataclasses import dataclass, field

from torchtitan.components.lr_scheduler import LRSchedulersContainer


def _wsds_multiplier(
    current_step: int,
    *,
    warmup_steps: int,
    adjusted_period_lengths: list[int],
    cum_period_end: list[int],
    decays: list[int],
    period_lr_multipliers: list[float] | None,
) -> float:
    """Return the LambdaLR multiplier on the peak LR for a given step.

    ``current_step`` is LambdaLR's 0-indexed ``last_epoch``; OLMo-core's WSD-S is
    1-indexed (matching torchtitan's default lambda, which applies the same +1).
    Warmup is subtracted from period 0 only; the per-period decay length uses the
    ORIGINAL (pre-warmup-adjustment) period lengths, exactly as OLMo-core does.
    """
    step = current_step + 1

    def peak(pidx: int) -> float:
        return 1.0 if period_lr_multipliers is None else period_lr_multipliers[pidx]

    if warmup_steps > 0 and step < warmup_steps:
        return peak(0) * step / warmup_steps

    adjusted = step - warmup_steps
    if adjusted >= cum_period_end[-1]:
        return 0.0

    pidx = next(i for i, end in enumerate(cum_period_end) if adjusted <= end)
    start = 0 if pidx == 0 else cum_period_end[pidx - 1]
    length = adjusted_period_lengths[pidx]
    pos = min(max(adjusted - start, 0), length)

    decay = decays[pidx]
    stable = length - decay
    if pos < stable or decay == 0:
        return peak(pidx)
    return peak(pidx) * (decay - (pos - stable)) / decay


class WSDSScheduler(LRSchedulersContainer):
    """WSD-S scheduler. Periods are expressed in steps by the planner."""

    @dataclass(kw_only=True, slots=True)
    class Config(LRSchedulersContainer.Config):
        period_lengths: list[int] = field(default_factory=list)
        """Incremental length of each Chinchilla period, in steps."""

        decay_fraction: float = 0.1
        """Fraction of each period spent linearly decaying the LR to 0."""

        period_lr_multipliers: list[float] | None = None
        """Optional per-period peak-LR scale (the stepped-schedule variant)."""

        # pyrefly: ignore [bad-override]
        def build(self, *, optimizers, training_steps):
            del training_steps  # the period table defines the schedule length
            periods = list(self.period_lengths)
            if not periods:
                raise ValueError("WSDSScheduler requires non-empty period_lengths.")
            warmup = int(self.warmup_steps)
            # Warmup is consumed from the first period only (OLMo-core).
            adjusted = [periods[0] - warmup] + periods[1:]
            cum_period_end = list(itertools.accumulate(adjusted))
            decays = [round(self.decay_fraction * p) for p in periods]
            lr_lambda = functools.partial(
                _wsds_multiplier,
                warmup_steps=warmup,
                adjusted_period_lengths=adjusted,
                cum_period_end=cum_period_end,
                decays=decays,
                period_lr_multipliers=self.period_lr_multipliers,
            )
            return WSDSScheduler(optimizers, lr_lambda)
