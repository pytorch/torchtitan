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
from vllm.v1.request import RequestStatus

from torchtitan.config import Configurable
from torchtitan.rl.generator.vllm.cpu_kv_offload import CPUKVOffload


class VLLMSessionKVHolder(Configurable):
    """Keeps each multi-turn session's reusable KV between its turns: held on GPU
    (``hold_gpu_blocks``), and with ``cpu_kv_offload`` in host RAM, where ended sessions go first.

    Between turns (a tool call, an env step) a session has no request in vLLM, so its blocks sit in
    vLLM's free queue, where any new request can take them; the next turn then recomputes its
    history. The holder takes a reference on them instead:

        turn 1 prefill done  ->  hold the blocks turn 2 would hit (attention + Gated-DeltaNet state)
        tool call            ->  held: no other request can take them
        turn 2 prefill done  ->  hold the blocks turn 3 would hit, drop turn 1's hold
        rollout ends         ->  release: the blocks go back to vLLM's free queue, still cached

    It holds at prefill, not at the turn's end: in vLLM's hybrid "align" mode the prompt-end GDN
    state leaves the request when decode starts.

    When memory runs short, it releases whole idle sessions, idle longest first:
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

    With ``cpu_kv_offload``, a session's CPU chunks are evicted first when it ends (`release`), except
    the chunks a live session also uses:

        CPU eviction queue, evicted first on the left; GRPO siblings s1, s2 share prompt chunk p
            [churn, p, s1, s2]
            release(session_ids=["s1"])  ->  [s1, churn, p, s2]   p stays: s2 still uses it
            release(session_ids=["s2"])  ->  [s2, p, s1, churn]

    Example:

        holder = VLLMSessionKVHolder.Config(cpu_kv_offload=CPUKVOffload.Config(size_gib=96)).build()
        engine_kwargs.update(holder.engine_kwargs())  # before the engine build
        engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
        holder.attach(engine)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        holder.track(internal_id, session_id="group=3/rollout=0", group_id=3)
        engine.step()
        holder.after_step()  # holds new prefixes, then releases idle sessions for the next step
        holder.release(session_ids=["group=3/rollout=0"])  # the rollout ended
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        hold_gpu_blocks: bool = True
        """Keep each session's reusable prefix on GPU between its turns. False needs
        ``cpu_kv_offload``: no GPU blocks are held, and the next turn reloads its history from CPU
        (an 8k-token turn on 1 H100: ~24 ms, vs ~16 ms GPU hit and ~196 ms recomputed)."""

        cpu_kv_offload: CPUKVOffload.Config | None = None
        """vLLM's native CPU KV offload (see `CPUKVOffload`). None: off."""

        def __post_init__(self):
            if not self.hold_gpu_blocks and self.cpu_kv_offload is None:
                raise ValueError(
                    "session_kv_holder with hold_gpu_blocks=False does nothing without "
                    "cpu_kv_offload: set cpu_kv_offload, or drop session_kv_holder"
                )

    def __init__(self, config: Config) -> None:
        self._hold_gpu_blocks = config.hold_gpu_blocks
        self.cpu_kv_offload = (
            config.cpu_kv_offload.build() if config.cpu_kv_offload is not None else None
        )
        # Set by `attach`: this rank's vLLM scheduler and its KV cache manager's parts.
        self._scheduler: Any = None
        self._coordinator: Any = None
        self._block_pool: Any = None
        self._num_kv_cache_groups = 0
        self.free_floor_blocks = 0
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

    def engine_kwargs(self) -> dict[str, Any]:
        """vLLM engine args the holder needs: the CPU offload connector's, if on."""
        return (
            {} if self.cpu_kv_offload is None else self.cpu_kv_offload.engine_kwargs()
        )

    def attach(self, engine: Any) -> None:
        """Read this rank's vLLM scheduler once the engine is built, and attach the CPU offload."""
        scheduler = engine.engine_core.engine_core.scheduler
        if self._hold_gpu_blocks:
            # Partial-block prefix hits or speculative decoding move the replay point away from the
            # prompt-end block this holds.
            if scheduler.kv_cache_manager.coordinator.enable_partial_hash_hits:
                raise ValueError(
                    "session_kv_holder does not support partial-block prefix hits"
                )
            if scheduler.vllm_config.speculative_config is not None:
                raise ValueError(
                    "session_kv_holder does not support speculative decoding"
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
        if self.cpu_kv_offload is not None:
            self.cpu_kv_offload.attach(engine)

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Once this request's prefill finishes, track its session (and hold its prefix)."""
        self._live[internal_request_id] = (session_id, group_id, False)

    def after_step(self) -> None:
        """After each engine step: track the sessions of requests whose prefill finished (and hold
        their prefix), mark sessions whose request left, and release idle sessions so the next
        step has room."""
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
                if self.cpu_kv_offload is not None:
                    self._session_hashes[session_id] = request.block_hashes
                if self._hold_gpu_blocks:
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
                # GRPO siblings share prompt blocks, so one release may free none.
                if self._block_pool.get_num_free_blocks() > free_before:
                    break
        self._evict_until_free()

    def _evict_until_free(self) -> None:
        """Release sessions with no running request, idle longest first, until enough blocks are free.

        A session whose request is running is skipped: that request still references most of its
        blocks, so releasing it frees little. A session whose next request is still waiting is
        released: its hold is all that keeps those blocks, and keeping it once memory is full stops
        every waiting request from starting.
        """
        if not self._hold_gpu_blocks:
            return
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
        for session_id in list(self._held):
            if self._block_pool.get_num_free_blocks() >= floor:
                return
            if session_id not in running_sessions:
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
        if self.cpu_kv_offload is not None:
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
            self.cpu_kv_offload.evict_first(
                block_hash
                for block_hash in ended_hashes
                if block_hash not in live_hashes
            )

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
