# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission: when the inter-generator router starts a new rollout group, and on which generator."""

from __future__ import annotations

import bisect
import itertools
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.rl.generator.engine import KVCacheBudget

logger = logging.getLogger(__name__)

# R averages over this window; 5 to 30 minutes changed simulated throughput by at most 3%.
GROWTH_WINDOW_S = 900.0
# Every session counts as a max-length context until this many sessions have ended: the first
# sessions to end are the shortest, so an R learned from a few of them is too low.
MIN_ENDED_SESSIONS = 50


class KVGrowthEstimateAdmission(Configurable):
    """Starts a new rollout group only while its generator's KV cache, times `limit`, holds every
    placed group at its expected final size, so multi-turn sessions keep their history cached.

    The router's calls, for group 7 of 16 rollouts; each group runs on one generator::

        policy.place({7: 16})  # -> {7: 1}, or {}: keep waiting
        policy.set_session_tokens(7, session_id="group=7/rollout=0", num_tokens=5000)  # call starts
        policy.set_session_tokens(7, session_id="group=7/rollout=0", num_tokens=6200)  # it returns
        policy.release_session("group=7/rollout=0")  # the rollout ends
        policy.release_group(7)  # the group ends

    `place` tries a generator charging nothing, else the one with the most room:

        it charges nothing                                   placed, even a group over the limit
        its charge + the group's <= limit x its KV blocks    placed
        its charge + the group's >  limit x its KV blocks    waits

    What each session of a placed group charges its generator, in KV blocks:

        live, grew g blocks since its first call:  its blocks + max(mean(G - g) over G > g, R - g)
        not started yet:                           the live sessions' mean first call + R
        until 50 sessions have ended:              a max-length context, started or not

        G = total growth of each session that ended in the last 15 min
        R = one session's growth: blocks added / sessions ended in those 15 min (Little's law)
        R - g covers sessions that outgrow the ended ones while lengths still rise
        no session charges more than a max-length context, whatever R says while it ramps

    Example (limit 1.0, 150 blocks per generator, R = 10, ended sessions grew G = 4, 10 or 16)::

        generator 0, two live 20-block sessions:
            grew 0  (first call 20):   20 + mean(4, 10, 16) = 30
            grew 12 (first call 8):    20 + (16 - 12)       = 24
        new group of 2:   54 + 2 x ((20 + 8) / 2 + 10) = 102 <= 150     placed

    Args:
        budgets: Each generator's KV cache size, by generator index.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        limit: float = 1.0
        """Placed groups' expected final KV blocks may reach `limit` x each generator's KV cache.
            1.0:        every placed session fits at its expected final size at once
            above 1.0:  more sessions; histories get evicted if they reach their final size together
        On hybrid models (Qwen3.5) vLLM also caches a Mamba state at every block boundary, so an
        idle session can hold 2-4x the blocks `KVCacheBudget.session_blocks` counts, more for longer
        sessions. Lower `limit` there; the first GPU run on a Qwen3.5 recipe sets the value.
        KV usage settles near limit / 2 when sessions grow far past their first call."""

    def __init__(self, config: Config, *, budgets: list[KVCacheBudget]):
        self.config = config
        # The generators share one config, so one budget's per-session cost fits all.
        self._budget = budgets[0]
        self._max_session_blocks = self._budget.session_blocks(
            self._budget.max_model_len
        )
        self.limits = [int(config.limit * budget.num_blocks) for budget in budgets]
        self._groups: dict[int, _PlacedGroup] = {}
        # The group of each live session, so `release_session` needs only the session id.
        self._session_group_ids: dict[str, int] = {}

        # As of the last `charges()`: what a session not started yet charges, R (None until 50
        # sessions have ended), and the mean first call of the live sessions (kept while none is
        # live).
        self._new_session_blocks = float(self._max_session_blocks)
        self.growth_per_session: float | None = None
        self._mean_first_call = 0.0

        # (time, blocks) for each growth of a live session, their sum, and (time, total growth) of
        # each session that ended.
        self._growth_events: deque[tuple[float, int]] = deque()
        self._growth_sum = 0
        self._ended: deque[tuple[float, int]] = deque()
        self._num_ended = 0
        logger.info(
            "KV admission: groups may charge %s KV cache blocks per generator; %s",
            self.limits,
            self._budget,
        )

    def place(self, group_sizes: dict[int, int]) -> dict[int, int]:
        """Place waiting groups, lowest id first, until one has to wait; return each placed
        group's generator.

        Example (one generator with room for 2 groups)::

            policy.place({5: 16, 3: 16, 9: 16})  # -> {3: 0, 5: 0}; group 9 keeps waiting

        Args:
            group_sizes: Rollouts per waiting group, by group id; each rollout opens one session.
        """
        # TODO: also cap new groups at a tenth of the pool per 5 s, spreading a cold
        # start's first prefills over ~50 s. Add it if a run shows that burst costs anything: this
        # gate already charges each group its final size when it is placed.
        charged = self.charges()
        placements: dict[int, int] = {}
        for group_id in sorted(group_sizes):
            generator = max(
                range(len(self.limits)),
                key=lambda index: (
                    charged[index] == 0,
                    self.limits[index] - charged[index],
                ),
            )
            # TODO: an env that opens several sessions per rollout (e.g. Verifiers sub-agents, or
            # two-player self-play) books too few here; learn sessions per group from finished
            # groups.
            new_charge = group_sizes[group_id] * self._new_session_blocks
            if (
                charged[generator] > 0
                and charged[generator] + new_charge > self.limits[generator]
            ):
                break
            self._groups[group_id] = _PlacedGroup(
                generator=generator, expected_sessions=group_sizes[group_id]
            )
            # What `charges()` would now return: the group's sessions have not started.
            charged[generator] += new_charge
            placements[group_id] = generator
        return placements

    def set_session_tokens(
        self, group_id: int, *, session_id: str, num_tokens: int
    ) -> None:
        """Record a session's context length: its prompt when a call starts, prompt plus
        completion when it returns."""
        group = self._groups[group_id]
        blocks = self._budget.session_blocks(num_tokens)
        session = group.sessions.get(session_id)
        if session is None:
            if session_id in group.seen_sessions:
                # A call that returned after its rollout released the session.
                return
            group.sessions[session_id] = _Session(
                first_call_blocks=blocks, blocks=blocks
            )
            group.seen_sessions.add(session_id)
            self._session_group_ids[session_id] = group_id
        else:
            grown = blocks - session.blocks
            if grown > 0:
                self._growth_events.append((time.monotonic(), grown))
                self._growth_sum += grown
            session.blocks = blocks

    def release_session(self, session_id: str) -> None:
        """Stop charging a session whose rollout ended, and learn its total growth. A session
        that never made a call is unknown here, so its group charges it until `release_group`."""
        group_id = self._session_group_ids.pop(session_id, None)
        if group_id is None:
            return
        session = self._groups[group_id].sessions.pop(session_id)
        self._ended.append(
            (time.monotonic(), session.blocks - session.first_call_blocks)
        )
        self._num_ended += 1

    def release_group(self, group_id: int) -> None:
        """Forget a finished group; its sessions still live count as ended."""
        group = self._groups.pop(group_id)
        now = time.monotonic()
        for session_id, session in group.sessions.items():
            del self._session_group_ids[session_id]
            # Sessions whose rollout did not release them, e.g. one that raised, end here.
            self._ended.append((now, session.blocks - session.first_call_blocks))
            self._num_ended += 1

    def summary(self) -> str:
        """Return this policy's state for the router's per-step log line."""
        fractions = [
            round(charged / limit, 2)
            for charged, limit in zip(self.charges(), self.limits)
        ]
        growth_text = (
            "cold start"
            if self.growth_per_session is None
            else f"growth {self.growth_per_session:.0f} blocks per session"
        )
        return (
            f"{len(self._groups)} groups placed, {growth_text}, "
            f"charged / limit per generator: {fractions}"
        )

    def charges(self) -> list[float]:
        """Return the KV blocks each generator's placed groups charge now, by generator index.
        Also updates R (`growth_per_session`) and what a session not started yet charges."""
        growth_per_session = self._growth_per_session(time.monotonic())
        remaining = _remaining_growth([total for _, total in self._ended])
        live_sessions = [
            session
            for group in self._groups.values()
            for session in group.sessions.values()
        ]
        if live_sessions:
            self._mean_first_call = sum(
                session.first_call_blocks for session in live_sessions
            ) / len(live_sessions)
        self._new_session_blocks = float(self._max_session_blocks)
        if growth_per_session is not None:
            self._new_session_blocks = min(
                self._mean_first_call + growth_per_session, self._new_session_blocks
            )
        self.growth_per_session = growth_per_session

        charged = [0.0] * len(self.limits)
        for group in self._groups.values():
            for session in group.sessions.values():
                charge = self._max_session_blocks
                if growth_per_session is not None:
                    grown = session.blocks - session.first_call_blocks
                    expected = max(remaining(grown), growth_per_session - grown)
                    charge = min(session.blocks + expected, charge)
                charged[group.generator] += charge
            not_started = max(0, group.expected_sessions - len(group.seen_sessions))
            charged[group.generator] += not_started * self._new_session_blocks
        return charged

    def _growth_per_session(self, now: float) -> float | None:
        """Return R, the blocks a session adds over its whole life; None until 50 sessions have
        ended.

        Example: in the last 15 min sessions added 9,000 blocks and 300 sessions ended -> 30.
        """
        cutoff = now - GROWTH_WINDOW_S
        while self._growth_events and self._growth_events[0][0] < cutoff:
            self._growth_sum -= self._growth_events.popleft()[1]
        while self._ended and self._ended[0][0] < cutoff:
            self._ended.popleft()
        if self._num_ended < MIN_ENDED_SESSIONS:
            return None
        # No session ended in the last 15 min: keep the last estimate.
        if not self._ended:
            return self.growth_per_session
        # Little's law: blocks added per second / sessions ended per second = one session's growth.
        return self._growth_sum / len(self._ended)


def _remaining_growth(ended_growth: list[int]) -> Callable[[int], float]:
    """Return a function from a live session's growth so far to its expected remaining growth:
    the mean of G - g over the ended sessions' growths G above g, or 0 if none is above.

    Example:

        remaining = _remaining_growth([4, 10, 16])
        remaining(0)   # -> 10.0, mean(4, 10, 16)
        remaining(12)  # -> 4.0, only 16 is above: 16 - 12
        remaining(16)  # -> 0.0, none is above
    """
    sorted_growth = sorted(ended_growth)
    # suffix[i] = sum(sorted_growth[i:])
    suffix = list(itertools.accumulate(reversed(sorted_growth)))[::-1] + [0]

    def remaining(grown: int) -> float:
        i = bisect.bisect_right(sorted_growth, grown)
        above = len(sorted_growth) - i
        return suffix[i] / above - grown if above else 0.0

    return remaining


@dataclass(kw_only=True, slots=True)
class _PlacedGroup:
    """A placed rollout group's sessions."""

    generator: int

    expected_sessions: int
    """Sessions the group is expected to open."""

    sessions: dict[str, _Session] = field(default_factory=dict)
    """Live sessions, by routing session id."""

    seen_sessions: set[str] = field(default_factory=set)
    """Sessions that have made a call; the rest of `expected_sessions` have not started."""


@dataclass(kw_only=True, slots=True)
class _Session:
    """A live session's KV blocks."""

    first_call_blocks: int
    """Blocks at its first call."""

    blocks: int
    """Blocks for its context so far."""
