# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hold each multi-turn session's reusable KV between its turns."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from vllm.utils.math_utils import cdiv

from torchtitan.config import Configurable


class VLLMSessionKVHolder(Configurable):
    """Keeps each multi-turn session's cached prefix on GPU between its turns.

    Between turns (a tool call, an env step) a session has no request in vLLM, so its blocks sit in
    vLLM's free queue, where any new request can take them; the next turn then recomputes its
    history. The holder takes a reference on them instead:

        turn 1 prefill done  ->  hold the blocks turn 2 would hit (attention + Gated-DeltaNet state)
        tool call            ->  held: no other request can take them
        turn 2 prefill done  ->  hold the blocks turn 3 would hit, drop turn 1's hold
        rollout ends         ->  release: the blocks go back to vLLM's free queue, still cached

    It holds at prefill, not at the turn's end: in vLLM's hybrid "align" mode the prompt-end GDN
    state leaves the request when decode starts.

    When memory runs short, it releases whole sessions, idle longest first and running ones last:
    - after each step, while fewer than ``floor`` blocks are free: the next step needs room, and the
      holder can't act inside a step;
    - after a step where nothing runs and requests wait: vLLM refused the head for lack of blocks.

    The floor is vLLM's watermark plus the most blocks the next step can add to running requests
    (vLLM schedules them before it admits new ones):

        free_floor_blocks = watermark_blocks
            + per KV cache group: cdiv(max_num_scheduled_tokens, block_size)  # one prefill chunk
        floor = free_floor_blocks
            + num_kv_cache_groups x len(running)  # a block each at a block boundary

    - Held blocks count as used in vLLM's ``kv_cache_usage``.
    - Every mutating method runs on every rank's engine thread in the same order, so the TP ranks'
      schedulers stay identical.

    Example:

        holder = VLLMSessionKVHolder.Config().build(scheduler=scheduler)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        holder.track(internal_id, session_id="group=3/rollout=0", group_id=3)
        engine.step()
        holder.after_step()  # holds new prefixes, then releases sessions for the next step
        holder.release(session_ids=["group=3/rollout=0"])  # the rollout ended

    Args:
        scheduler: The vLLM v1 scheduler of this rank's in-process engine.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config, *, scheduler: Any) -> None:
        del config
        # Partial-block prefix hits or speculative decoding move the replay point away from the
        # prompt-end block this holds.
        if scheduler.kv_cache_manager.coordinator.enable_partial_hash_hits:
            raise ValueError(
                "session_kv_holder does not support partial-block prefix hits"
            )
        if scheduler.vllm_config.speculative_config is not None:
            raise ValueError("session_kv_holder does not support speculative decoding")
        self._scheduler = scheduler
        kv_cache_manager = scheduler.kv_cache_manager
        self._coordinator = kv_cache_manager.coordinator
        self._block_pool = kv_cache_manager.block_pool
        self._num_kv_cache_groups = kv_cache_manager.num_kv_cache_groups
        self.free_floor_blocks = kv_cache_manager.watermark_blocks + sum(
            cdiv(scheduler.max_num_scheduled_tokens, group.kv_cache_spec.block_size)
            for group in kv_cache_manager.kv_cache_config.kv_cache_groups
        )
        # Requests in vLLM: internal request id -> (session, group, prefill seen).
        self._live: dict[str, tuple[str, int, bool]] = {}
        # Held blocks per session, idle longest first.
        self._held: OrderedDict[str, list[Any]] = OrderedDict()
        self._session_groups: dict[str, int] = {}
        self._group_sessions: dict[int, set[str]] = {}
        # Each session's latest request's block hashes (vLLM appends decode blocks to this list):
        # the prefix its next turn can reuse, until `release`.
        self._session_hashes: dict[str, list[bytes]] = {}
        # Sessions with stored hashes, per group: an eviction drops the session from
        # `_group_sessions`, so a group's `release` needs this to drop their hashes.
        self._hashed_sessions: dict[int, set[str]] = {}
        # Each tracked request's vLLM prefill stats (its hits at the first schedule), taken at
        # `track`: vLLM detaches them at the first output, before `after_step` runs.
        self._prefill_stats: dict[str, Any] = {}
        # Plain ints, so other threads (e.g. a logging endpoint) can read them safely.
        self.num_sessions = 0
        self.num_evicted = 0
        self.num_stall_evicted = 0
        # A later turn's prompt tokens that earlier turns computed: hit on GPU, or recomputed.
        self.gpu_hit_prefix_tokens = 0
        self.recomputed_prefix_tokens = 0

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Hold this request's prefix once its prefill finishes."""
        self._live[internal_request_id] = (session_id, group_id, False)
        self._prefill_stats[internal_request_id] = self._scheduler.requests[
            internal_request_id
        ].prefill_stats

    def after_step(self) -> None:
        """After each engine step: hold the prefix of requests whose prefill finished, mark sessions
        whose request left, and release held sessions (idle first) so the next step has room."""
        for internal_id, (session_id, group_id, held) in list(self._live.items()):
            request = self._scheduler.requests.get(internal_id)
            if request is None:  # finished or aborted: the session is idle from now on
                del self._live[internal_id]
                self._prefill_stats.pop(internal_id, None)
                if session_id in self._held:
                    self._held.move_to_end(session_id)
            elif not held and request.num_computed_tokens >= request.num_prompt_tokens:
                self._live[internal_id] = (session_id, group_id, True)
                prefill_stats = self._prefill_stats.pop(internal_id)
                previous_hashes = self._session_hashes.get(session_id)
                if previous_hashes is not None:
                    self._count_prefix_tokens(request, previous_hashes, prefill_stats)
                self._session_hashes[session_id] = request.block_hashes
                self._hashed_sessions.setdefault(group_id, set()).add(session_id)
                self._hold(request, session_id, group_id)

        # Nothing runs and requests wait: with an empty batch vLLM skips the watermark, so it refused
        # the head for lack of blocks (or the batch just drained: one cheap extra release).
        scheduler = self._scheduler
        if not scheduler.running and (scheduler.waiting or scheduler.skipped_waiting):
            free_before = self._block_pool.get_num_free_blocks()
            for session_id in list(self._held):
                self._release(session_id)
                self.num_evicted += 1
                self.num_stall_evicted += 1
                # GRPO siblings share prompt blocks, so one release may free none.
                if self._block_pool.get_num_free_blocks() > free_before:
                    break
        self._evict_until_free()

    def _evict_until_free(self) -> None:
        """Release held sessions until enough blocks are free, those with a running request last.

        - Sessions with no running request go first, idle longest first. A session whose next
          request is still waiting is released too: its hold is all that keeps those blocks, and
          keeping it once memory is full stops every waiting request from starting.
        - Sessions whose request is running go last, oldest hold first. A release frees only the
          blocks that request no longer uses: its prompt-end GDN state (dropped once it decodes)
          and the extra copy of a block that `_hold` may take. Skipping them under heavy pressure
          makes vLLM preempt running requests instead.
        """
        running = self._scheduler.running
        # GRPO siblings decode in lockstep, so all running requests can cross a boundary together.
        floor = self.free_floor_blocks + self._num_kv_cache_groups * len(running)
        if self._block_pool.get_num_free_blocks() >= floor:
            return
        running_sessions = {
            self._live[request.request_id][0]
            for request in running
            if request.request_id in self._live
        }
        # False (no running request) sorts before True, and sorted() is stable: sessions with no
        # running request go first, then running ones, each part in `_held` order.
        for session_id in sorted(self._held, key=running_sessions.__contains__):
            if self._block_pool.get_num_free_blocks() >= floor:
                return
            self._release(session_id)
            self.num_evicted += 1

    def release(self, session_ids: list[str] = (), group_ids: list[int] = ()) -> None:
        """Release sessions that make no more calls, and every session of finished groups."""
        released = set(session_ids)
        for group_id in group_ids:
            released |= self._group_sessions.pop(group_id, set())
            for session_id in self._hashed_sessions.pop(group_id, ()):
                self._session_hashes.pop(session_id, None)
        # Sorted, not set order: string hashes differ per process, and every TP rank must free
        # blocks in the same order.
        for session_id in sorted(released):
            self._release(session_id)
            self._session_hashes.pop(session_id, None)
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

    def _count_prefix_tokens(
        self, request: Any, previous_hashes: list[bytes], prefill_stats: Any
    ) -> None:
        """Split the prompt tokens this session's earlier turns computed into hit on GPU and recomputed.

        Example, 16-token blocks:

            turn 1  prompt 50 + completion 20 tokens   ->  4 full blocks: 64 tokens computed
            turn 2  prompt = those 70 tokens + 3       ->  shares the 4 blocks: 64 reusable
                    vLLM hits 48                       ->  gpu_hit 48, recomputed 16
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
        self.recomputed_prefix_tokens += max(
            0, reusable_tokens - prefill_stats.num_cached_tokens
        )
