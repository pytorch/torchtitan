# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hold each multi-turn session's reusable KV between its turns; ended sessions' CPU copies are
evicted first."""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any

from vllm.utils.math_utils import cdiv
from vllm.v1.request import RequestStatus


class SessionKVHolder:
    """Keeps a session's reusable prefix out of vLLM's free pool until its next turn or its end.

    vLLM frees a finished turn's blocks to an LRU free queue, where any new request can evict them;
    the session's next turn then recomputes its history. In vLLM's hybrid "align" mode the prompt-end
    Gated-DeltaNet state even leaves the request when decode starts. So right after a request's
    prefill, this holds the blocks vLLM's own prefix lookup says a continuation would hit (attention
    blocks and the matching GDN state blocks), and lets go of the session's previous hold. Held blocks
    count as used in vLLM's ``kv_cache_usage``.

    It releases whole idle sessions, idle longest first:
    - before each step, while fewer than ``floor`` blocks are free (it can't act inside a step);
    - after a step where nothing runs and requests wait: vLLM refused the head for lack of blocks.

    The floor is vLLM's watermark plus the most blocks the next step can add to running requests
    (vLLM schedules them before it admits new ones):

        free_floor_blocks = watermark_blocks
            + per KV cache group: cdiv(max_num_scheduled_tokens, block_size)  # one prefill chunk
        floor = free_floor_blocks
            + num_kv_cache_groups x len(running)  # a block each at a block boundary

    Too small, and vLLM preempts a running request (a recompute, not a hang); too big only holds
    fewer sessions.

    A release only demotes: the blocks go to vLLM's LRU free queue and stay prefix hits until reused.
    Every mutating method must run on the engine thread of every rank, in the same order, so the
    schedulers of all TP ranks stay identical.

    With CPU offload, vLLM also copies every computed block to CPU. When a session ends
    (`release`), its CPU chunks are evicted first, except the chunks a live session also uses:

        CPU eviction queue, evicted first on the left; GRPO siblings s1, s2 share prompt chunk p
            [churn, p, s1, s2]
            release(session_ids=["s1"])  ->  [s1, churn, p, s2]   p stays: s2 still uses it
            release(session_ids=["s2"])  ->  [s2, p, s1, churn]

    Example:

        holder = SessionKVHolder(scheduler)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        holder.track(internal_id, session_id="group=3/rollout=0", group_id=3)
        holder.evict_until_free()  # before each engine.step()
        engine.step()
        holder.after_step()  # holds the prefix of requests whose prefill just finished
        holder.release(session_ids=["group=3/rollout=0"])  # the rollout ended

    Args:
        scheduler: The vLLM v1 scheduler of this rank's in-process engine.
        hold: Hold sessions' GPU blocks. False: only ended sessions' CPU chunks are evicted first.
        offload_policy: vLLM's CPU offload `LRUCachePolicy`; None without CPU offload.
    """

    def __init__(
        self,
        scheduler: Any,
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
        kv_cache_manager = scheduler.kv_cache_manager
        self._coordinator = kv_cache_manager.coordinator
        self._block_pool = kv_cache_manager.block_pool
        self._num_kv_cache_groups = kv_cache_manager.num_kv_cache_groups
        self.free_floor_blocks = kv_cache_manager.watermark_blocks + sum(
            cdiv(scheduler.max_num_scheduled_tokens, group.kv_cache_spec.block_size)
            for group in kv_cache_manager.kv_cache_config.kv_cache_groups
        )
        self._hold_enabled = hold
        self._offload_policy = offload_policy
        # Requests in vLLM: internal request id -> (session, group, prefill seen).
        self._live: dict[str, tuple[str, int, bool]] = {}
        # Held blocks per session, idle longest first.
        self._held: OrderedDict[str, list[Any]] = OrderedDict()
        # Sessions past their first prefill, until `release`.
        self._session_groups: dict[str, int] = {}
        self._group_sessions: dict[int, set[str]] = {}
        # Each session's latest request's block hashes (vLLM appends decode blocks to this list):
        # the prefix its next turn can reuse, and with offload the CPU chunks `release` evicts first.
        # If a turn re-renders the history, chunks only an earlier turn had are not evicted first.
        self._session_hashes: dict[str, list[bytes]] = {}
        # Each tracked request's vLLM prefill stats (its hits at the first schedule), taken at
        # `track`: vLLM detaches them at the first output, before `after_step` runs.
        self._prefill_stats: dict[str, Any] = {}
        # Sessions holding each block: GRPO siblings hold their shared prompt blocks once each.
        self._held_block_refs: Counter[int] = Counter()
        # Plain ints, so other threads (e.g. a logging endpoint) can read them safely.
        self.num_sessions = 0
        self.num_held_blocks = 0
        self.num_evicted = 0
        self.num_stall_evicted = 0
        # A session's later turns: prompt tokens its earlier turns computed, by where they came from.
        self.gpu_hit_prefix_tokens = 0
        self.cpu_hit_prefix_tokens = 0
        self.recomputed_prefix_tokens = 0

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Once this request's prefill finishes, track its session (and with `hold`, its prefix)."""
        self._live[internal_request_id] = (session_id, group_id, False)
        self._prefill_stats[internal_request_id] = self._scheduler.requests[
            internal_request_id
        ].prefill_stats

    def after_step(self) -> None:
        """Track the sessions of requests whose prefill finished (with `hold`, hold their prefix);
        mark sessions whose request left; release sessions if vLLM could not admit anything."""
        for internal_id, (session_id, group_id, prefill_seen) in list(
            self._live.items()
        ):
            request = self._scheduler.requests.get(internal_id)
            if request is None:  # finished or aborted: the session is idle from now on
                del self._live[internal_id]
                self._prefill_stats.pop(internal_id, None)
                if session_id in self._held:
                    self._held.move_to_end(session_id)
            elif (
                not prefill_seen
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                self._live[internal_id] = (session_id, group_id, True)
                prefill_stats = self._prefill_stats.pop(internal_id)
                previous_hashes = self._session_hashes.get(session_id)
                if previous_hashes is not None:
                    self._count_prefix_tokens(request, previous_hashes, prefill_stats)
                self._session_groups[session_id] = group_id
                self._group_sessions.setdefault(group_id, set()).add(session_id)
                self._session_hashes[session_id] = request.block_hashes
                if self._hold_enabled:
                    self._hold(request, session_id)

        # Nothing runs and requests wait: with an empty batch vLLM skips the watermark, so it refused
        # the head for lack of blocks (or the batch just drained: one cheap extra release).
        # A request loading its KV from CPU waits for the load, not for blocks. One whose CPU lookup
        # is deferred (a chunk still being stored) still counts: at worst one release per empty step.
        scheduler = self._scheduler
        if not scheduler.running and any(
            request.status != RequestStatus.WAITING_FOR_REMOTE_KVS
            for request in (*scheduler.skipped_waiting, *scheduler.waiting)
        ):
            free_before = self._block_pool.get_num_free_blocks()
            for session_id in list(self._held):
                self._release(session_id)
                self.num_evicted += 1
                self.num_stall_evicted += 1
                # GRPO siblings share prompt blocks, so one release may free none.
                if self._block_pool.get_num_free_blocks() > free_before:
                    break

    def evict_until_free(self) -> None:
        """Release sessions with no running request, idle longest first, until enough blocks are free.

        A session whose request is running is skipped: that request still references most of its
        blocks, so releasing it frees little. A session whose next request is still waiting is
        released: its hold is all that keeps those blocks, and keeping it once memory is full stops
        every waiting request from starting.
        """
        if not self._hold_enabled:
            return
        running_ids = {request.request_id for request in self._scheduler.running}
        # GRPO siblings decode in lockstep, so all running requests can cross a boundary together.
        floor = self.free_floor_blocks + self._num_kv_cache_groups * len(running_ids)
        if self._block_pool.get_num_free_blocks() >= floor:
            return
        running = {self._live[i][0] for i in running_ids if i in self._live}
        for session_id in list(self._held):
            if self._block_pool.get_num_free_blocks() >= floor:
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
        ended_hashes = [
            block_hash
            for session_id in released_in_order
            for block_hash in self._session_hashes.pop(session_id, ())
        ]

        # ======== CPU offload: the ended sessions' chunks are evicted first ========
        if self._offload_policy is not None:
            from vllm.v1.kv_offload.base import make_offload_key

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

    def cpu_used_fraction(self) -> float:
        """Fraction of the CPU tier's chunks that hold KV (stored or being stored)."""
        return len(self._offload_policy.chunks) / self._offload_policy.cache_capacity

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
        self._held_block_refs.update(block.block_id for block in blocks)
        self.num_held_blocks = len(self._held_block_refs)
        self.num_sessions += 1

    def _release(self, session_id: str) -> None:
        blocks = self._held.pop(session_id, None)
        if blocks is not None:
            self.num_sessions -= 1
            self._block_pool.free_blocks(reversed(blocks))
            for block in blocks:
                self._held_block_refs[block.block_id] -= 1
                if not self._held_block_refs[block.block_id]:
                    del self._held_block_refs[block.block_id]
            self.num_held_blocks = len(self._held_block_refs)

    def _count_prefix_tokens(
        self, request: Any, previous_hashes: list[bytes], prefill_stats: Any
    ) -> None:
        """Split the prompt tokens this session's earlier turns computed by where this turn got them.

        Example, 16-token blocks:

            turn 1  prompt 50 + completion 20 tokens   ->  4 full blocks: 64 tokens computed
            turn 2  prompt = those 70 tokens + 3       ->  shares the 4 blocks: 64 reusable
                    vLLM hits 48 on GPU, 0 from CPU    ->  gpu_hit 48, cpu_hit 0, recomputed 16
        """
        num_shared_blocks = 0
        for previous_hash, block_hash in zip(previous_hashes, request.block_hashes):
            if previous_hash != block_hash:
                break
            num_shared_blocks += 1
        shared_tokens = num_shared_blocks * self._block_pool.hash_block_size
        # vLLM hits at most num_prompt_tokens - 1 tokens, in whole scheduler blocks.
        alignment = self._coordinator.scheduler_block_size
        reusable_tokens = (
            min(shared_tokens, request.num_prompt_tokens - 1) // alignment * alignment
        )
        self.gpu_hit_prefix_tokens += prefill_stats.num_local_cached_tokens
        self.cpu_hit_prefix_tokens += prefill_stats.num_external_cached_tokens
        self.recomputed_prefix_tokens += max(
            0, reusable_tokens - prefill_stats.num_cached_tokens
        )
