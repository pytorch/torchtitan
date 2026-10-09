# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hold each multi-turn session's reusable KV between its turns."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class SessionKVHolder:
    """Keeps a session's reusable prefix out of vLLM's free pool until its next turn or its end.

    vLLM frees a finished turn's blocks to an LRU free queue, where any new request can evict them;
    the session's next turn then recomputes its history. In vLLM's hybrid "align" mode the prompt-end
    Gated-DeltaNet state even leaves the request when decode starts. So right after a request's
    prefill, this holds the blocks vLLM's own prefix lookup says a continuation would hit (attention
    blocks and the matching GDN state blocks), and lets go of the session's previous hold. Held blocks
    count as used in vLLM's ``kv_cache_usage``.

    When free blocks fall below ``free_floor_blocks``, whole sessions without a running request are
    released, the one idle longest first. Every mutating method must run on the engine thread of
    every rank, in the same order, so the schedulers of all TP ranks stay identical.

    Example:

        holder = SessionKVHolder(scheduler, free_floor_blocks=400)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        holder.track(internal_id, session_id="group=3/rollout=0", group_id=3)
        holder.evict_until_free()  # before each engine.step()
        engine.step()
        holder.after_step()  # holds the prefix of requests whose prefill just finished
        holder.release(session_ids=["group=3/rollout=0"])  # the rollout ended

    Args:
        scheduler: The vLLM v1 scheduler of this rank's in-process engine.
        free_floor_blocks: Release idle sessions while fewer blocks than this are free.
    """

    def __init__(self, scheduler: Any, free_floor_blocks: int) -> None:
        self._scheduler = scheduler
        self._coordinator = scheduler.kv_cache_manager.coordinator
        self._block_pool = scheduler.kv_cache_manager.block_pool
        self.free_floor_blocks = free_floor_blocks
        # Requests in vLLM: internal request id -> (session, group, prefill seen).
        self._live: dict[str, tuple[str, int, bool]] = {}
        # Held blocks per session, idle longest first.
        self._held: OrderedDict[str, list[Any]] = OrderedDict()
        self._session_groups: dict[str, int] = {}
        self._group_sessions: dict[int, set[str]] = {}
        # Plain ints, so other threads (e.g. a logging endpoint) can read them safely.
        self.num_sessions = 0
        self.num_evicted = 0

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Hold this request's prefix once its prefill finishes."""
        self._live[internal_request_id] = (session_id, group_id, False)

    def after_step(self) -> None:
        """Hold the prefix of requests whose prefill finished; mark sessions whose request left."""
        for internal_id, (session_id, group_id, held) in list(self._live.items()):
            request = self._scheduler.requests.get(internal_id)
            if request is None:  # finished or aborted: the session is idle from now on
                del self._live[internal_id]
                if session_id in self._held:
                    self._held.move_to_end(session_id)
            elif not held and request.num_computed_tokens >= request.num_prompt_tokens:
                self._live[internal_id] = (session_id, group_id, True)
                self._hold(request, session_id, group_id)

    def evict_until_free(self) -> None:
        """Release sessions with no running request, idle longest first, until enough blocks are free.

        A session whose request is running is skipped: that request still references most of its
        blocks, so releasing it frees little. A session whose next request is still waiting is
        released: its hold is all that keeps those blocks, and keeping it once memory is full stops
        every waiting request from starting.
        """
        if self._block_pool.get_num_free_blocks() >= self.free_floor_blocks:
            return
        running_ids = {request.request_id for request in self._scheduler.running}
        running = {self._live[i][0] for i in running_ids if i in self._live}
        for session_id in list(self._held):
            if self._block_pool.get_num_free_blocks() >= self.free_floor_blocks:
                return
            if session_id not in running:
                self._release(session_id)
                self.num_evicted += 1

    def release(self, session_ids: list[str] = (), group_ids: list[int] = ()) -> None:
        """Release sessions that make no more calls, and every session of finished groups."""
        released = set(session_ids)
        for group_id in group_ids:
            released |= self._group_sessions.pop(group_id, set())
        # Sorted, not set order: string hashes differ per process, and every TP rank must free
        # blocks in the same order.
        for session_id in sorted(released):
            self._release(session_id)
        # A request still before its prefill must not re-hold a released session.
        dropped_groups = set(group_ids)
        for internal_id, (session_id, group_id, held) in list(self._live.items()):
            if session_id in released or group_id in dropped_groups:
                self._live[internal_id] = (session_id, group_id, True)

    def _hold(self, request: Any, session_id: str, group_id: int) -> None:
        # The prompt's last token is recomputed for its logits, so a continuation hits at most
        # num_prompt_tokens - 1 tokens; the lookup returns the attention blocks and, per GDN
        # group, the state block that ends on that aligned boundary. vLLM returns the oldest
        # cached copy of each block, so the hold may take a duplicate of a block this request's
        # table also owns (about one extra block per session while it decodes).
        hit_blocks, _, _ = self._coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_prompt_tokens - 1
        )
        blocks = [b for kv_group in hit_blocks for b in kv_group if not b.is_null]
        # Take the new hold before dropping the old one, so shared prefix blocks stay held.
        self._block_pool.touch(blocks)
        self._release(session_id)
        self._held[session_id] = blocks
        self._session_groups[session_id] = group_id
        self._group_sessions.setdefault(group_id, set()).add(session_id)
        self.num_sessions += 1

    def _release(self, session_id: str) -> None:
        blocks = self._held.pop(session_id, None)
        group_id = self._session_groups.pop(session_id, None)
        if group_id is not None:
            self._group_sessions.get(group_id, set()).discard(session_id)
        if blocks is not None:
            self.num_sessions -= 1
            self._block_pool.free_blocks(reversed(blocks))
