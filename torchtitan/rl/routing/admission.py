# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission: which generator a new rollout group starts on, and when, by KV room."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.rl.generator.engine import KVCacheCapacity

# A group is placed while its generator's charged KV, the group's included, stays under this
# share of the generator's blocks and state slots. The rest absorbs placed sessions' growth.
_ADMIT_FRACTION = 0.95
# The charge of a session that has not called, or whose call runs, uses the means of the last
# this many first prompts and replies.
_NUM_RECENT_CALLS = 64


class KVRoomAdmission(Configurable):
    """Places new rollout groups on generators by KV room, and counts the KV each placed
    session holds or will hold. Engines can't count it: they count an idle session's cached
    KV as free.

    The router's calls, for group 7 of 2 rollouts:

        admission.place(7, num_sessions=2)  # -> 1, the generator with the most room; None: wait
        admission.start_call("r0", session_id="7/0", group_id=7, num_prompt_tokens=4000)  # -> 1
        admission.end_call("r0", num_reply_tokens=500)
        admission.release_session("7/0")  # -> 1, the generator that holds its KV
        admission.release_groups([7])

    What each session of a placed group charges (`KVCacheCapacity.session_blocks`):

        a call of it runs      its latest prompt + the mean reply, running
        idle between calls     its last prompt + that call's reply, idle
        has not called yet     the mean first prompt + the mean reply, running

    The means are over the last 64 first prompts and replies, 0 before any. A group fits a
    generator that has no placed group, or, once a first prompt was seen, one where

        charged blocks + the group's  <= 95% x num_blocks
        charged states + the group's  <= 95% x num_state_slots   (when the engine caps states)

    Args:
        capacities: Each generator's KV cache capacity, by generator index.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config, *, capacities: Sequence[KVCacheCapacity]):
        self.config = config
        self._capacities = list(capacities)
        self._groups: dict[int, _PlacedGroup] = {}
        self._session_groups: dict[str, int] = {}
        # (group id, session id) of each running call, by request id.
        self._running_calls: dict[str, tuple[int, str]] = {}
        self._first_prompts: deque[int] = deque(maxlen=_NUM_RECENT_CALLS)
        self._replies: deque[int] = deque(maxlen=_NUM_RECENT_CALLS)

    def place(self, group_id: int, *, num_sessions: int) -> int | None:
        """Place a new group on the generator with the most block room left after it, if one
        fits; return that generator's index, or None."""
        num_new_session_tokens = _mean(self._first_prompts) + _mean(self._replies)
        best, best_room = None, None
        for index, (capacity, charge) in enumerate(
            zip(self._capacities, self._charges(), strict=True)
        ):
            room = (
                _ADMIT_FRACTION * capacity.num_blocks
                - charge.num_blocks
                - num_sessions
                * capacity.session_blocks(num_new_session_tokens, running=True)
            )
            state_room = (
                None
                if capacity.num_state_slots is None
                else _ADMIT_FRACTION * capacity.num_state_slots
                - charge.num_states
                - num_sessions * capacity.num_running_states
            )
            # Before any first prompt, the charge of a group that has not called is 0.
            fits = charge.num_groups == 0 or (
                bool(self._first_prompts)
                and room >= 0
                and (state_room is None or state_room >= 0)
            )
            if fits and (best_room is None or room > best_room):
                best, best_room = index, room

        if best is not None:
            self._groups[group_id] = _PlacedGroup(
                generator=best, num_sessions=num_sessions
            )
        return best

    def start_call(
        self,
        request_id: str,
        *,
        session_id: str | None,
        group_id: int,
        num_prompt_tokens: int,
    ) -> int:
        """Charge a starting call to its session; return its group's generator."""
        group = self._groups.get(group_id)
        if group is None or session_id is None:
            raise ValueError(
                f"call {request_id!r} of group {group_id}, session {session_id!r}: admission "
                "needs a routing session id and a group that acquire_group placed and "
                "release_groups did not release"
            )

        session = group.sessions.get(session_id)
        if session is None:
            group.num_started += 1
            self._first_prompts.append(num_prompt_tokens)
            self._session_groups[session_id] = group_id
            session = group.sessions[session_id] = _Session()
        session.num_tokens = num_prompt_tokens
        session.latest_request_id = request_id
        session.running_request_ids.add(request_id)
        self._running_calls[request_id] = (group_id, session_id)
        return group.generator

    def end_call(self, request_id: str, *, num_reply_tokens: int) -> None:
        """Record an ended call's reply; a call of a released session or group is ignored."""
        call = self._running_calls.pop(request_id, None)
        if call is None:
            return
        group_id, session_id = call
        group = self._groups.get(group_id)
        session = None if group is None else group.sessions.get(session_id)
        if session is None:
            return

        self._replies.append(num_reply_tokens)
        session.running_request_ids.discard(request_id)
        # An older call that ends late leaves the newest call's prompt as the history.
        if request_id == session.latest_request_id:
            session.num_tokens += num_reply_tokens

    def release_session(self, session_id: str) -> int | None:
        """Stop charging a session whose rollout ended; return the generator that holds its KV,
        or None if no placed group has it. It still counts as started."""
        group_id = self._session_groups.pop(session_id, None)
        if group_id is None:
            return None
        group = self._groups[group_id]
        del group.sessions[session_id]
        return group.generator

    def release_groups(self, group_ids: Sequence[int]) -> None:
        """Stop charging finished groups and their sessions; unknown ids are skipped."""
        for group_id in group_ids:
            group = self._groups.pop(group_id, None)
            if group is None:
                continue
            for session_id in group.sessions:
                del self._session_groups[session_id]

    def charged_fractions(self) -> list[float]:
        """Per generator, the larger of charged blocks over its blocks and charged states over
        its state slots."""
        fractions = []
        for capacity, charge in zip(self._capacities, self._charges(), strict=True):
            fraction = charge.num_blocks / capacity.num_blocks
            if capacity.num_state_slots is not None:
                fraction = max(fraction, charge.num_states / capacity.num_state_slots)
            fractions.append(fraction)
        return fractions

    def _charges(self) -> list[_Charge]:
        """Each generator's placed groups, and the blocks and states they charge."""
        mean_reply = _mean(self._replies)
        num_new_session_tokens = _mean(self._first_prompts) + mean_reply
        charges = [_Charge() for _ in self._capacities]
        for group in self._groups.values():
            capacity = self._capacities[group.generator]
            charge = charges[group.generator]
            charge.num_groups += 1
            for session in group.sessions.values():
                running = bool(session.running_request_ids)
                num_tokens = session.num_tokens + (mean_reply if running else 0)
                charge.num_blocks += capacity.session_blocks(
                    num_tokens, running=running
                )
                charge.num_states += capacity.session_states(running=running)

            # Sessions that have not called yet; retried attempts can open more than the group.
            num_waiting = max(0, group.num_sessions - group.num_started)
            charge.num_blocks += num_waiting * capacity.session_blocks(
                num_new_session_tokens, running=True
            )
            charge.num_states += num_waiting * capacity.num_running_states
        return charges


@dataclass(kw_only=True, slots=True)
class _Session:
    """One started session of a placed group."""

    num_tokens: int = 0
    """Its latest call's prompt, plus that call's reply once it ended."""

    latest_request_id: str = ""

    running_request_ids: set[str] = field(default_factory=set)
    """Its calls that started and have not ended; a retry can overlap the call it repeats."""


@dataclass(kw_only=True, slots=True)
class _PlacedGroup:
    """A placed rollout group: every call of its sessions goes to `generator`."""

    generator: int
    num_sessions: int
    """Sessions the group opens: its size."""

    num_started: int = 0
    """Sessions that called, released ones included."""

    sessions: dict[str, _Session] = field(default_factory=dict)
    """Started sessions not released, by routing session id."""


@dataclass(kw_only=True, slots=True)
class _Charge:
    """What one generator's placed groups charge."""

    num_groups: int = 0
    num_blocks: int = 0
    num_states: int = 0


def _mean(values: deque[int]) -> int:
    return sum(values) // len(values) if values else 0
