# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hold each multi-turn session's reusable KV between its turns; ended sessions' CPU copies are
evicted first."""

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

    With CPU offload, vLLM also copies every computed block to CPU. When a session ends
    (`release`), its CPU chunks are evicted first, except the chunks a live session also uses:

        CPU eviction queue, evicted first on the left; GRPO siblings s1, s2 share prompt chunk p
            [churn, p, s1, s2]
            release(session_ids=["s1"])  ->  [s1, churn, p, s2]   p stays: s2 still uses it
            release(session_ids=["s2"])  ->  [s2, p, s1, churn]

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
        hold: Hold sessions' GPU blocks. False: only ended sessions' CPU chunks are evicted first.
        offload_policy: vLLM's CPU offload `LRUCachePolicy`; None without CPU offload.
    """

    def __init__(
        self,
        scheduler: Any,
        free_floor_blocks: int,
        hold: bool = True,
        offload_policy: Any | None = None,
    ) -> None:
        if offload_policy is not None:
            # Imported here: runs without CPU offload never load vLLM's offload code.
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

            # `release` reorders the LRU's private `evictable_chunks` (pinned vLLM version).
            if not isinstance(offload_policy, LRUCachePolicy):
                raise RuntimeError(
                    "CPU KV offload needs vLLM's LRUCachePolicy, got "
                    f"{type(offload_policy).__name__}"
                )
        self._scheduler = scheduler
        self._coordinator = scheduler.kv_cache_manager.coordinator
        self._block_pool = scheduler.kv_cache_manager.block_pool
        self.free_floor_blocks = free_floor_blocks
        self._hold_enabled = hold
        self._offload_policy = offload_policy
        self._num_kv_cache_groups = scheduler.kv_cache_manager.num_kv_cache_groups
        # Requests in vLLM: internal request id -> (session, group, prefill seen).
        self._live: dict[str, tuple[str, int, bool]] = {}
        # Held blocks per session, idle longest first.
        self._held: OrderedDict[str, list[Any]] = OrderedDict()
        # Sessions past their first prefill, until `release`.
        self._session_groups: dict[str, int] = {}
        self._group_sessions: dict[int, set[str]] = {}
        # With offload: each session's latest request's block hashes. Its prompt holds the earlier
        # turns, and vLLM appends decode blocks to this list. If a turn re-renders the history,
        # chunks only an earlier turn had are not evicted first.
        self._session_hashes: dict[str, list[bytes]] = {}
        # Plain ints, so other threads (e.g. a logging endpoint) can read them safely.
        self.num_sessions = 0
        self.num_evicted = 0

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Once this request's prefill finishes, track its session (and with `hold`, its prefix)."""
        self._live[internal_request_id] = (session_id, group_id, False)

    def after_step(self) -> None:
        """Track the sessions of requests whose prefill finished (with `hold`, hold their prefix);
        mark sessions whose request left."""
        for internal_id, (session_id, group_id, prefill_seen) in list(
            self._live.items()
        ):
            request = self._scheduler.requests.get(internal_id)
            if request is None:  # finished or aborted: the session is idle from now on
                del self._live[internal_id]
                if session_id in self._held:
                    self._held.move_to_end(session_id)
            elif (
                not prefill_seen
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                self._live[internal_id] = (session_id, group_id, True)
                self._session_groups[session_id] = group_id
                self._group_sessions.setdefault(group_id, set()).add(session_id)
                if self._offload_policy is not None:
                    self._session_hashes[session_id] = request.block_hashes
                if self._hold_enabled:
                    self._hold(request, session_id)

    def evict_until_free(self) -> None:
        """Release sessions with no running request, idle longest first, until enough blocks are free.

        A session whose request is running is skipped: that request still references most of its
        blocks, so releasing it frees little. A session whose next request is still waiting is
        released: its hold is all that keeps those blocks, and keeping it once memory is full stops
        every waiting request from starting.
        """
        if (
            not self._hold_enabled
            or self._block_pool.get_num_free_blocks() >= self.free_floor_blocks
        ):
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
        dropped_groups = set(group_ids)
        for group_id in group_ids:
            released |= self._group_sessions.pop(group_id, set())
        # Sorted, not set order: string hashes differ per process, and every TP rank must free
        # blocks in the same order.
        released_in_order = sorted(released)
        for session_id in released_in_order:
            self._release(session_id)
            group_id = self._session_groups.pop(session_id, None)
            if group_id is not None:
                self._group_sessions.get(group_id, set()).discard(session_id)

        # ======== CPU offload: the ended sessions' chunks are evicted first ========
        if self._offload_policy is not None:
            from vllm.v1.kv_offload.base import make_offload_key

            ended_hashes = [
                block_hash
                for session_id in released_in_order
                for block_hash in self._session_hashes.pop(session_id, ())
            ]
            # Chunks a live session uses keep their place: sessions past their first prefill,
            # and requests still before it (e.g. a GRPO sibling's first turn).
            # TODO: rebuilt on every release: 0.6 ms for 250 live Qwen3.5 sessions of 16k tokens,
            # 23 ms with 16-token hash blocks (attention-only models). A per-hash refcount,
            # caught up as sessions grow, would cost only the released sessions' length.
            live_hashes = set().union(*self._session_hashes.values())
            for internal_id, (session_id, group_id, prefill_seen) in self._live.items():
                request = self._scheduler.requests.get(internal_id)
                ended = session_id in released or group_id in dropped_groups
                if not prefill_seen and not ended and request is not None:
                    live_hashes.update(request.block_hashes)
            # One key per block and KV cache group. Keys vLLM never stored, or is still storing
            # or loading, are skipped.
            # TODO: a chunk whose store finishes after the release (the last turn's final blocks,
            # stored on the next engine step) keeps its LRU place; evict it first once stored.
            evictable_chunks = self._offload_policy.evictable_chunks
            for block_hash in ended_hashes:
                if block_hash in live_hashes:
                    continue
                for group_idx in range(self._num_kv_cache_groups):
                    key = make_offload_key(block_hash, group_idx)
                    if key in evictable_chunks:
                        evictable_chunks.move_to_end(key, last=False)

        # A request still before its prefill must not re-hold a released session.
        for internal_id, (session_id, group_id, prefill_seen) in list(
            self._live.items()
        ):
            if session_id in released or group_id in dropped_groups:
                self._live[internal_id] = (session_id, group_id, True)

    def _hold(self, request: Any, session_id: str) -> None:
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
        self.num_sessions += 1

    def _release(self, session_id: str) -> None:
        blocks = self._held.pop(session_id, None)
        if blocks is not None:
            self.num_sessions -= 1
            self._block_pool.free_blocks(reversed(blocks))
