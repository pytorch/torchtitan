# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Demand rule of the adaptive rollout buffer: how many prompt groups may be in the pipeline.

Words
-----
group        one prompt with its N sampled answers. Everything here is counted in groups.
P            groups the trainer consumes per training step.
ready        finished groups waiting to be trained on (the shelf). The next batch of P is cut from them.
generating   groups still producing answers (waiting for a generator, or in flight).
demand       how many groups the buffer may hold at once: generating + ready + the batch being trained. The one
             number this rule decides, at every step start.
stall        the trainer arrives at a step start and fewer than P groups are ready: it waits.
rejected     a finished group whose answers all got the same reward; thrown away on arrival. The "untrainable
             share" is the fraction of finished groups rejected.
policy age   of a trained group: training steps between its generation and its training.

The rule
--------
At a step start, `demand - ready` of the slots hold groups that are not ready. Call that `unavailable`. The trainer
stalls when fewer than P groups are ready, so the demand that would have avoided a stall at that step is exactly
`unavailable + P`. Keep the last `lookback_steps` values of `unavailable` and set

    demand_needed = P + quantile(unavailable over the last lookback_steps, 1 - stall_probability) + 1

then move toward it: up by `damping_factor` of the gap, down by one group per step. `+ 1` is one spare group.
Nothing is assumed about how long groups take: the count already contains everything (slow groups, refills of
rejected groups, a workload getting slower).

Down is slow because a large ready shelf makes `unavailable` small while it lasts. Cutting the demand below the
groups already admitted stops admission; the trainer uses up the shelf, then stalls on groups admitted too late. A
demand above the mean-age ceiling (see below) still comes down by `damping_factor` of the gap to the ceiling.

`unavailable` is a sum of one "not finished yet" coin flip per group in the buffer, so it is close to normal whatever
the distribution of generation times. The quantile is read off a normal fitted to the window, `mean + t x sd`,
where `t` is the Student-t prediction multiplier for the next value given `lookback_steps` samples (1.92 for 10
samples at 95%). It uses every point of the window and can exceed the largest value seen.

Demand never exceeds the mean-age ceiling: by Little's law the buffer may hold A batches of trainable groups plus
the batch in training, plus the slots the rejected groups occupy while they generate,

    ceiling = A * P + P + generating * untrainable_share

with A = `target_offpolicy_steps` if given, else `max_offpolicy_steps` if given; with neither there is no ceiling. When
the ceiling binds the trainer may stall rather than train on older data; a log line says so.

Example (P = 8, lookback 10, stall probability 5%):

    unavailable over the last 10 step starts = [44, 46, 41, 48, 45, 47, 50, 43, 46, 45]
    mean 45.5, sd 2.55, t = 1.92                 ->  quantile = 45.5 + 1.92 x 2.55 = 50.4 -> 51
    demand_needed = 8 + 51 + 1 = 60
    demand was 56 -> half the gap up: 56 + ceil(4 x 0.5) = 58
    demand was 64 -> one group down: 63
"""

import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StallDrivenDemand:
    """Demand = P + quantile of the unavailable count + 1, up a share of the gap or down one group per step, under a ceiling.

    Args:
        num_prompts_per_train_step: P, groups the trainer consumes per step.
        max_offpolicy_steps: A, the hard limit on policy age; the buffer drops groups older than this. Without a
            target below, it is also the mean age the ceiling holds under. None: no hard limit.
        target_offpolicy_steps: optional. If given, the MEAN policy age is held at or under it, at the cost of
            stalling: demand is capped at the mean-age ceiling for this age. None: the ceiling is applied at
            `max_offpolicy_steps`, if given; with neither there is no ceiling.
        lookback_steps: how many of the most recent step starts the quantile is taken over; older values are forgotten.
        stall_probability: the share of steps allowed to stall; the quantile is taken at 1 - stall_probability. This
            is what the rule aims for when nothing else limits the demand. It cannot be met when the age ceiling or
            the generators' capacity caps the demand below what the workload needs, and it assumes some finished
            groups are trainable.
        start_batches: demand at the first step, in batches of P; the rule learns the rest from the run itself.
        damping_factor: how much of the gap between the current demand and `demand_needed` is closed per step going
            up, or coming down to the mean-age ceiling; otherwise demand comes down one group per step. 0.5 moves
            half the way each step; 1.0 jumps straight there.

    The name: the rule is driven by what the trainer would have stalled on. `demand - ready` at a step start is the
    number of groups that were not there for it; the rule sizes the buffer for the value that number stays under.

    Example:
        demand = StallDrivenDemand(num_prompts_per_train_step=8, max_offpolicy_steps=10)
        demand.demand                                                       # -> 24: three batches to start
        demand.observe(step=1, ready=0, generating=24, completed=0, trainable=0)
        # -> 29: unavailable 24, quantile 24 (one sample), needed 8 + 24 + 1 = 33, half the gap up from 24
    """

    num_prompts_per_train_step: int
    max_offpolicy_steps: int | None = None
    target_offpolicy_steps: int | None = None
    lookback_steps: int = 10
    stall_probability: float = 0.05
    start_batches: int = 3
    damping_factor: float = 0.5
    demand: int = field(init=False)
    # "ok", or "age-limited" while the ceiling binds
    state: str = field(default="ok", init=False)
    _unavailable: deque = field(init=False)
    _generating: deque = field(init=False)
    _completed: deque = field(init=False)
    _trainable: deque = field(init=False)

    def __post_init__(self) -> None:
        self.demand = self.start_batches * self.num_prompts_per_train_step
        self._unavailable = deque(maxlen=self.lookback_steps)
        self._generating = deque(maxlen=self.lookback_steps)
        self._completed = deque(maxlen=self.lookback_steps)
        self._trainable = deque(maxlen=self.lookback_steps)

    def observe(
        self,
        *,
        step: int,
        ready: int,
        generating: int,
        completed: int,
        trainable: int,
    ) -> int:
        """Update the demand from one step start and return it.

        Args:
            step: Training step about to start (for the log line).
            ready: Finished groups not yet trained (finalized, selected, or queued for the trainer).
            generating: Groups still producing answers (waiting or in flight).
            completed: Groups that finished since the previous step start.
            trainable: Of those, groups with a learning signal.
        """
        P = self.num_prompts_per_train_step

        # Record what this step start looked like
        self._unavailable.append(max(0, self.demand - ready))
        self._generating.append(generating)
        self._completed.append(completed)
        self._trainable.append(trainable)

        # The value unavailable stays under with probability 1 - stall_probability, plus the batch and one spare group
        unavailable_upper_bound = estimate_next_unavailable_upper_bound(
            history=list(self._unavailable),
            probability=1 - self.stall_probability,
        )
        demand_needed = P + unavailable_upper_bound + 1

        # At most one group down per step, so a large shelf cannot stop admission; the ceiling below can cut faster
        demand_needed = max(demand_needed, self.demand - 1)

        # Cap the demand so the mean policy age stays under the target, or under the max if no target was given
        mean_age_limit = (
            self.target_offpolicy_steps
            if self.target_offpolicy_steps is not None
            else self.max_offpolicy_steps
        )
        if mean_age_limit is not None:
            ceiling = mean_age_ceiling(
                num_prompts_per_train_step=P,
                mean_age_limit=mean_age_limit,
                groups_generating=statistics.mean(self._generating),
                untrainable_share=1.0
                - sum(self._trainable) / max(1, sum(self._completed)),
            )
            demand_needed, self.state = clamp_demand_needed(
                demand_needed=demand_needed,
                cap=math.floor(ceiling),
                reason=f"to keep the mean policy age under {mean_age_limit} steps",
                step=step,
                previous_state=self.state,
            )

        # Move part of the way toward demand_needed
        self.demand = smooth_demand_toward_needed(
            current_demand=self.demand,
            demand_needed=demand_needed,
            damping_factor=self.damping_factor,
        )
        return self.demand


def estimate_next_unavailable_upper_bound(
    *,
    history: list[int],
    probability: float,
) -> int:
    """The value the NEXT entry stays under with `probability`, from a normal fitted to the history: mean + t x sd, rounded up.

    Example:
        estimate_next_unavailable_upper_bound(
            history=[44, 46, 41, 48, 45, 47, 50, 43, 46, 45],
            probability=0.95,
        )   # -> 51 (45.5 + 1.92 x 2.55)
        estimate_next_unavailable_upper_bound(
            history=[45, 45, 45],
            probability=0.95,
        )   # -> 45 (sd 0)
    """
    if len(history) < 2:
        return max(history)
    mean = statistics.mean(history)
    sd = statistics.stdev(history)
    margin = prediction_multiplier(
        samples=len(history),
        probability=probability,
    )
    return math.ceil(mean + margin * sd)


def prediction_multiplier(
    *,
    samples: int,
    probability: float,
) -> float:
    """Margin, in standard deviations, for predicting the next value of a normal variable whose mean and sd were
    estimated from `samples` values: the Student-t quantile with samples - 1 degrees of freedom, times sqrt(1 + 1 / samples).

    Example:
        prediction_multiplier(
            samples=10,
            probability=0.95,
        )   # -> 1.92   (t = 1.833, x sqrt(1.1))
        prediction_multiplier(
            samples=1000,
            probability=0.95,
        )   # -> 1.65   (the normal quantile)
    """
    z = statistics.NormalDist().inv_cdf(probability)
    dof = samples - 1
    # Student-t quantile from the normal quantile z (no library function for it without scipy). Cornish-Fisher series
    # in powers of 1 / dof:
    #   t = z + (z^3 + z) / (4 dof) + (5 z^5 + 16 z^3 + 3 z) / (96 dof^2) + (3 z^7 + 19 z^5 + 17 z^3 - 15 z) / (384 dof^3)
    # Table check: dof 9 at 95% -> 1.833 (exact 1.833); dof 4 at 95% -> 2.13 (exact 2.132).
    # Same as scipy.stats.t.ppf(probability, dof).
    t = (
        z
        + (z**3 + z) / (4 * dof)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * dof**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * dof**3)
    )
    return t * math.sqrt(1 + 1 / samples)


def mean_age_ceiling(
    *,
    num_prompts_per_train_step: int,
    mean_age_limit: int,
    groups_generating: float,
    untrainable_share: float,
) -> float:
    """Most groups the buffer may hold while keeping the MEAN policy age under `mean_age_limit`.

    By Little's law, mean age = (groups in the buffer that will be trained) / (groups trained per step). So the buffer
    may hold `mean_age_limit` batches of trainable groups, plus the batch in training. Rejected groups also hold slots
    while they generate but never train, so their share of the generating count is added on top:

        ceiling = mean_age_limit * P  +  P  +  generating * untrainable_share

    Example:
        mean_age_ceiling(
            num_prompts_per_train_step=8,
            mean_age_limit=4,
            groups_generating=40,
            untrainable_share=0.36,
        )   # -> 32 + 8 + 14.4 = 54.4
    """
    return (
        mean_age_limit * num_prompts_per_train_step
        + num_prompts_per_train_step
        + groups_generating * untrainable_share
    )


def clamp_demand_needed(
    *,
    demand_needed: int,
    cap: int,
    reason: str,
    step: int,
    previous_state: str,
) -> tuple[int, str]:
    """Cap the demand the rule asked for; returns (capped demand, state) and logs when the cap starts or stops binding.

    `state` is "age-limited" while the cap binds, "ok" otherwise. `reason` completes the sentence "capped at N
    <reason>" in the log, so the reader knows which limit is binding.

    Example:
        clamp_demand_needed(
            demand_needed=70,
            cap=54,
            reason="to keep the mean policy age under 4 steps",
            step=12,
            previous_state="ok",
        )
        # -> (54, "age-limited") and logs: "step 12: buffer wants 70 groups but is capped at 54 to keep the mean
        #    policy age under 4 steps; the trainer may stall"
    """
    state = "age-limited" if demand_needed > cap else "ok"
    if state == "age-limited" and previous_state != "age-limited":
        logger.warning(
            f"step {step}: buffer wants {demand_needed} groups but is capped at {cap} {reason}; the trainer may stall"
        )
    elif state == "ok" and previous_state == "age-limited":
        logger.info(
            f"step {step}: buffer no longer capped ({demand_needed} groups needed, cap {cap})"
        )
    return min(demand_needed, cap), state


def smooth_demand_toward_needed(
    *,
    current_demand: int,
    demand_needed: int,
    damping_factor: float,
) -> int:
    """Next demand: move `damping_factor` of the gap from `current_demand` toward `demand_needed`, the same going up and down.

    Rounds away from `current_demand`, so a small gap is still closed instead of never.

    Example:
        smooth_demand_toward_needed(
            current_demand=64,
            demand_needed=59,
            damping_factor=0.5,
        )   # -> 61   (gap -5, move -3)
        smooth_demand_toward_needed(
            current_demand=64,
            demand_needed=73,
            damping_factor=0.5,
        )   # -> 69   (gap +9, move +5)
    """
    gap = demand_needed - current_demand
    if gap > 0:
        return current_demand + math.ceil(gap * damping_factor)
    return current_demand - math.ceil(-gap * damping_factor)
