# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evict ended multi-turn sessions' KV from vLLM's prefix cache."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TrackedRequest:
    """One request's block hashes, counted while they may still be reused."""

    internal_request_id: str
    block_hashes: list[BlockHash]
    """vLLM's `Request.block_hashes`, which vLLM extends in place as the reply fills blocks."""
    num_counted: int = 0
    """How many of `block_hashes` are counted in `_num_requests_by_hash`."""


class VLLMSessionKVEvictor:
    """Evicts the cached KV of ended multi-turn sessions from vLLM's prefix cache, so new
    requests reuse it before the cached history of sessions that wait between turns.

    Each block hash counts the requests that may still reuse it: every live session's newest
    request, and any replaced or released request still in vLLM (e.g. preempted). At count 0,
    the hash's free blocks lose it and move to the front of vLLM's free queue:

        session A ended              -> A's own blocks: count 0 -> evicted, reused first
        session B waits on a tool    -> B's blocks: count 1     -> stay cached
        a system prompt A and B share -> count 1 (B)            -> stays cached

    - Blocks in use (a running request, a KV connector's save) keep their hash and stay in
      the prefix cache. A block in use when its hash is evicted goes in the first
      `after_step` after it is free, unless a request counts its hash again.
    - Every TP rank makes the same calls in the same order and evicts in block id order, so
      the ranks' schedulers stay identical.
    - With CPU offload, the evicted hashes also go to the connector, once the saves pending
      at their eviction finish, so their CPU blocks are evicted first too.

    Example:

        evictor = VLLMSessionKVEvictor(scheduler=scheduler)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        evictor.track(internal_id, session_id="group=3/rollout=0", group_id=3)
        engine.step()
        evictor.after_step()
        evictor.release(session_ids=["group=3/rollout=0"], group_ids=[])  # the rollout ended

    Args:
        scheduler: The vLLM v1 scheduler of this rank's in-process engine.
        offload_connector: The CPU offload connector, with ``get_pending_save_ids()`` and
            ``release_session_cache(released_hashes, active_hashes)``; None without one.
    """

    def __init__(self, *, scheduler: Any, offload_connector: Any | None = None) -> None:
        # TODO: vLLM's session eviction hint (vllm-project/vllm#51428) would replace this
        #   class's reads and edits of BlockPool internals.
        kv_cache_manager = scheduler.kv_cache_manager
        # A group block larger than the hash block is cached under its last hash, which
        # `Request.block_hashes` holds, so eviction reaches it. Partial hits only over-evict:
        # a live session prefills at most the block its history ends in, so warn, don't raise.
        if kv_cache_manager.coordinator.enable_partial_hash_hits:
            logger.warning(
                "vLLM caches partial-block prefix hits (a Mamba group's block is larger than "
                "the hash block). Evicting a block also drops its partial hits, so a live "
                "session whose history ends inside it prefills that block again."
            )
        self._scheduler = scheduler
        self._block_pool = kv_cache_manager.block_pool
        self._kv_cache_group_ids = range(kv_cache_manager.num_kv_cache_groups)
        # Each live session's newest request.
        self._sessions: dict[str, _TrackedRequest] = {}
        self._group_sessions: dict[int, set[str]] = {}
        # Replaced or released requests still in vLLM, by internal request id.
        self._ended_requests: dict[str, _TrackedRequest] = {}
        self._num_requests_by_hash: dict[BlockHash, int] = {}
        # Hashes whose eviction skipped a block in use; `after_step` retries them.
        self._held_block_hashes: set[BlockHash] = set()
        # CPU release hints: hashes evicted since the last hint, and the batch whose hint
        # waits for the saves that were pending when it formed.
        self._offload_connector = offload_connector
        self._cpu_hashes_to_release: set[BlockHash] = set()
        self._cpu_releasing_hashes: set[BlockHash] = set()
        self._cpu_releasing_save_ids: set[int] = set()
        # A plain int, so other threads (e.g. a logging endpoint) can read it safely.
        self.num_evicted_blocks = 0

    @property
    def num_sessions(self) -> int:
        """Live sessions on this engine."""
        return len(self._sessions)

    @property
    def has_pending_work(self) -> bool:
        """Whether a CPU release hint still waits for engine steps (its saves to finish)."""
        return bool(self._cpu_hashes_to_release or self._cpu_releasing_hashes)

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Make this request its session's history; evict what its prompt dropped from the
        previous request's (e.g. a chat template that strips old reasoning)."""
        # TODO: a request that reaches the engine after its session's or group's release (an
        #   abandoned attempt still in transit) re-creates the session, and nothing releases
        #   it: its hashes stay counted until vLLM's LRU reuses its blocks. Fix: skip tracking
        #   requests of recently released groups. Rare: only an abandoned attempt in transit.
        # TODO: counting the whole prompt costs ~0.17 ms per turn at 16-token blocks and
        #   32K-token histories (CPU bench). The previous request's hashes are a prefix of
        #   this list (chained hashes), so counting only the new part would cost O(new tokens).
        request = _TrackedRequest(
            internal_request_id=internal_request_id,
            block_hashes=self._scheduler.requests[internal_request_id].block_hashes,
        )
        self._count_new_block_hashes(request)
        previous = self._sessions.get(session_id)
        self._sessions[session_id] = request
        self._group_sessions.setdefault(group_id, set()).add(session_id)
        if previous is not None:
            self._forget([previous])

    def after_step(self) -> None:
        """Evict the blocks of replaced or released requests that left vLLM in this step, and
        the blocks that were in use at their eviction and are free now."""
        if self._held_block_hashes:
            # A hash a request counts again stays: that request's end evicts it.
            self._count_all_new_block_hashes()
            held_block_hashes = self._held_block_hashes
            self._held_block_hashes = set()
            self._evict(
                [
                    block_hash
                    for block_hash in held_block_hashes
                    if block_hash not in self._num_requests_by_hash
                ]
            )

        finished = [
            request
            for internal_id, request in self._ended_requests.items()
            if internal_id not in self._scheduler.requests
        ]
        for request in finished:
            del self._ended_requests[request.internal_request_id]
        if finished:
            self._forget(finished)
        self._release_cpu_cache()

    def release(self, *, session_ids: list[str], group_ids: list[int]) -> None:
        """Evict the KV of sessions that make no more calls, and of every session of finished
        groups. Unknown ids are ignored: another generator served them."""
        released = set(session_ids)
        for group_id in group_ids:
            released |= self._group_sessions.pop(group_id, set())
        # Sorted, not set order: string hashes differ per process, and every TP rank must
        # evict in the same order.
        self._forget(
            [
                self._sessions.pop(session_id)
                for session_id in sorted(released)
                if session_id in self._sessions
            ]
        )
        self._release_cpu_cache()

    def _forget(self, requests: list[_TrackedRequest]) -> None:
        """Uncount `requests` that left vLLM and evict their hashes that nothing counts; keep
        counting the others until `after_step` sees them leave."""
        ended_block_hashes: list[BlockHash] = []
        for request in requests:
            if request.internal_request_id in self._scheduler.requests:
                self._ended_requests[request.internal_request_id] = request
                continue
            self._count_new_block_hashes(request)
            for block_hash in request.block_hashes:
                num_requests = self._num_requests_by_hash.pop(block_hash) - 1
                if num_requests:
                    self._num_requests_by_hash[block_hash] = num_requests
            ended_block_hashes += request.block_hashes
        uncounted = [
            block_hash
            for block_hash in dict.fromkeys(ended_block_hashes)
            if block_hash not in self._num_requests_by_hash
        ]
        if not uncounted:
            return
        self._count_all_new_block_hashes()
        released = [
            block_hash
            for block_hash in uncounted
            if block_hash not in self._num_requests_by_hash
        ]
        self._evict(released)
        if self._offload_connector is not None:
            self._cpu_hashes_to_release.update(released)

    def _release_cpu_cache(self) -> None:
        """Send the evicted hashes to the CPU offload connector once the saves pending when
        they were batched finish; a hash a request counts again by then stays."""
        if not self.has_pending_work:
            return
        connector = self._offload_connector
        if not self._cpu_releasing_hashes:
            self._cpu_releasing_hashes = self._cpu_hashes_to_release
            self._cpu_hashes_to_release = set()
            # Freeze this batch's saves (finish-time saves included), so later saves can't
            # postpone its hint forever.
            self._cpu_releasing_save_ids = connector.get_pending_save_ids()
        if self._cpu_releasing_save_ids & connector.get_pending_save_ids():
            return
        self._count_all_new_block_hashes()
        active_hashes = set(self._num_requests_by_hash)
        released = self._cpu_releasing_hashes - active_hashes
        self._cpu_releasing_hashes = set()
        self._cpu_releasing_save_ids = set()
        if released:
            connector.release_session_cache(released, active_hashes)

    def _count_all_new_block_hashes(self) -> None:
        """Count every tracked request's new output: a preempted request's blocks are free and
        cached until it resumes, and an ended request may share them."""
        for request in (*self._sessions.values(), *self._ended_requests.values()):
            self._count_new_block_hashes(request)

    def _count_new_block_hashes(self, request: _TrackedRequest) -> None:
        for block_hash in request.block_hashes[request.num_counted :]:
            self._num_requests_by_hash[block_hash] = (
                self._num_requests_by_hash.get(block_hash, 0) + 1
            )
        request.num_counted = len(request.block_hashes)

    def _evict(self, block_hashes: list[BlockHash]) -> None:
        """Move the free blocks of `block_hashes` that nothing counts to the free queue's front,
        without their hash."""
        cached_blocks = self._block_pool.cached_block_hash_to_block
        free_block_queue = self._block_pool.free_block_queue
        evicted = []
        for block_hash in block_hashes:
            for kv_cache_group_id in self._kv_cache_group_ids:
                key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
                # vLLM may cache one hash in several blocks (siblings that prefilled one
                # prompt in the same step). Each lookup returns one; blocks in use are set
                # aside to reach the rest, then put back.
                in_use = []
                while (block := cached_blocks.get_one_block(key)) is not None:
                    if block.ref_cnt > 0:
                        # In use, e.g. by a KV connector's save or lazy copy, or a free that
                        # async scheduling defers: it keeps its hash until `after_step` retries.
                        cached_blocks.pop(key, block.block_id)
                        in_use.append(block)
                        self._held_block_hashes.add(block_hash)
                        continue
                    free_block_queue.remove(block)
                    # Drops the block's hashes, as vLLM does when it reuses a cached block.
                    self._block_pool.evict_blocks({block.block_id})
                    evicted.append(block)
                for block in in_use:
                    cached_blocks.insert(key, block)
        # Block id order, so every TP rank leaves the same free queue.
        free_block_queue.prepend_n(sorted(evicted, key=lambda block: block.block_id))
        self.num_evicted_blocks += len(evicted)
