# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evict cached KV that no live rollout will reuse from vLLM's prefix cache."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    get_block_hash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)

from torchtitan.rl.generator.types import KVCacheReading


@dataclass(slots=True)
class _Session:
    """One rollout's multi-turn conversation (a routing session) on this engine."""

    block_hashes: set[BlockHash] = field(default_factory=set)
    """Block hashes of the session's history: its newest request's prompt and output."""
    num_requests_in_flight: int = 0


@dataclass(slots=True)
class _TrackedRequest:
    """A request in vLLM, from admission until it finishes."""

    session_id: str
    block_hashes: list[BlockHash]
    """vLLM's ``Request.block_hashes``, which vLLM extends in place as output tokens fill blocks."""
    num_recorded_block_hashes: int = 0
    """How many of ``block_hashes`` the session has recorded."""
    session_released: bool = False
    """The rollout ended while this request ran; ``finish`` evicts its blocks that no live session
    records."""


class VLLMSessionKVEvictor:
    """Evicts the cached KV blocks of ended rollouts from vLLM's prefix cache, so new requests
    reuse them before the cached history of live rollouts.

    Each block hash counts the live sessions (rollouts) whose history holds it. When the count
    reaches 0, the hash's free blocks lose the hash and move to the front of vLLM's free queue:

        rollout A ended  -> A's own blocks: count 0 -> evicted (taken first by the next request)
        rollout B waits on a tool call -> B's blocks: count 1 -> stay cached
        a system prompt that A and B share -> count 1 (B) -> stays cached

    - Blocks a running request holds are never touched.
    - A preempted request's output is recorded before any eviction: its blocks are free and
      cached until it resumes.
    - A new request of a session with no request in flight replaces the session's history, so
      an edited history (a dropped reply) releases the dropped blocks.
    - Every TP rank makes the same calls in the same order (the engine loop's decisions) and
      evicts in block id order, so the ranks' schedulers stay identical.

    Example:

        evictor = VLLMSessionKVEvictor(scheduler=scheduler)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        evictor.track(
            internal_id,
            request_id="group=3/rollout=0/turn=2",
            session_id="group=3/rollout=0",
            group_id=3,
        )
        finished = engine.step()
        evictor.finish([output.request_id for output in finished])
        evictor.release(session_ids=["group=3/rollout=0"], group_ids=[])  # the rollout ended

    Args:
        scheduler: The vLLM v1 scheduler of this rank's in-process engine.
    """

    def __init__(self, *, scheduler: Any) -> None:
        kv_cache_manager = scheduler.kv_cache_manager
        managers = kv_cache_manager.coordinator.single_type_managers
        self._scheduler = scheduler
        self._block_pool = kv_cache_manager.block_pool
        self._kv_cache_group_ids = range(len(managers))
        # TODO: an upstream vLLM session eviction hint (vllm-project/vllm#51428) would replace
        # this class's reads and edits of BlockPool internals.
        num_block_pools = len({id(manager.block_pool) for manager in managers})
        if num_block_pools > 1:
            raise ValueError(
                "VLLMSessionKVEvictor needs every KV cache group in one vLLM block pool; "
                f"found {num_block_pools} block pools for {len(managers)} KV cache groups"
            )
        # A group block larger than the hash block makes vLLM cache partial blocks under extra
        # hashes (aliases), which this class does not track.
        hash_block_size = self._block_pool.hash_block_size
        block_sizes = [manager.block_size for manager in managers]
        if any(block_size != hash_block_size for block_size in block_sizes):
            raise ValueError(
                "VLLMSessionKVEvictor needs every KV cache group's block size to equal the "
                f"block pool's hash_block_size; found block sizes {block_sizes} and "
                f"hash_block_size {hash_block_size}"
            )
        # Requests in vLLM by TorchTitan request id, the id finished outputs carry.
        self._requests: dict[str, _TrackedRequest] = {}
        self._sessions: dict[str, _Session] = {}
        self._session_ids_by_group: dict[int, set[str]] = {}
        # Live sessions whose history holds each block hash.
        self._num_owners_by_hash: dict[BlockHash, int] = {}
        # `log_stats` reads it (and `num_sessions`) from the actor's thread; a plain int and a
        # `len` are safe to read there.
        self.num_evicted_blocks = 0

    @property
    def num_sessions(self) -> int:
        """Live sessions on this engine."""
        return len(self._sessions)

    def kv_cache_reading(self) -> KVCacheReading:
        """Read the KV cache. Once ended rollouts' blocks are evicted, every cached free block
        belongs to a live rollout, so occupancy counts it; vLLM's usage counts it as free.

        Walks the free queue: 0.3-0.7 ms per 10k free blocks on CPU.
        """
        block_pool = self._block_pool
        num_free_uncached_blocks = sum(
            block.block_hash is None
            for block in block_pool.free_block_queue.iter_blocks_after(None)
        )
        # TODO: on hybrid models (Qwen3.5) vLLM caches a Mamba state at every block boundary,
        # so a waiting rollout keeps several Mamba state blocks per layer group where its next
        # turn needs 1, and its cached blocks reach about twice its need. Trimming those states
        # is a follow-up.
        # The null block is never free; vLLM's usage leaves it out too.
        return KVCacheReading(
            occupancy=1.0 - num_free_uncached_blocks / (block_pool.num_gpu_blocks - 1),
            usage=block_pool.get_usage(),
        )

    def track(
        self,
        internal_request_id: str,
        *,
        request_id: str,
        session_id: str,
        group_id: int,
    ) -> None:
        """Record a newly admitted request's prompt under its session.

        Args:
            internal_request_id: vLLM's id for the request (TorchTitan's id plus a random suffix).
            request_id: TorchTitan's id, which the request's finished output carries.
            session_id: The rollout's routing session id.
            group_id: The rollout's group id.
        """
        request = self._scheduler.requests[internal_request_id]
        session = self._sessions.get(session_id)
        if session is None:
            session = self._sessions[session_id] = _Session()
            self._session_ids_by_group.setdefault(group_id, set()).add(session_id)
        elif session.num_requests_in_flight == 0:
            # The new prompt is the session's history from now on. Turn 1 that drops turn 0's
            # reply (a chat template that strips old reasoning) releases the reply's blocks.
            dropped = session.block_hashes.difference(request.block_hashes)
            session.block_hashes -= dropped
            self._evict(self._drop_owners(dropped))
        session.num_requests_in_flight += 1
        tracked = self._requests[request_id] = _TrackedRequest(
            session_id=session_id, block_hashes=request.block_hashes
        )
        # Record the prompt now, so the session's set is complete between evictions.
        self._record_new_block_hashes(tracked)

    def finish(self, request_ids: list[str]) -> None:
        """After each engine step: record the finished requests' output, and evict the blocks
        of requests whose rollout already ended."""
        ended_block_hashes: list[BlockHash] = []
        for request_id in request_ids:
            request = self._requests.pop(request_id)
            if request.session_released:
                ended_block_hashes += request.block_hashes
            else:
                self._record_new_block_hashes(request)
                self._sessions[request.session_id].num_requests_in_flight -= 1
        self._evict(ended_block_hashes)

    def release(self, session_ids: list[str], group_ids: list[int]) -> None:
        """Evict the blocks of ended rollouts, and of every rollout of ended groups, that no
        live session records. Unknown ids are ignored: another DP replica or generator served
        them."""
        released_ids = set(session_ids)
        for group_id in group_ids:
            released_ids |= self._session_ids_by_group.pop(group_id, set())
        released = {
            session_id: session
            for session_id in released_ids
            if (session := self._sessions.pop(session_id, None)) is not None
        }
        if not released:
            return
        # A request still in vLLM keeps its blocks until it finishes; `finish` evicts them then.
        # TODO: with a vLLM KV connector, a save still in flight also holds a dead block; evict it
        # once the save finishes.
        for request in self._requests.values():
            if request.session_id in released:
                request.session_released = True
        self._evict(
            self._drop_owners(
                block_hash
                for session in released.values()
                for block_hash in session.block_hashes
            )
        )

    def _record_new_block_hashes(self, request: _TrackedRequest) -> None:
        """Add the request's hashes that its session has not recorded yet, and count them."""
        recorded = self._sessions[request.session_id].block_hashes
        for block_hash in request.block_hashes[request.num_recorded_block_hashes :]:
            if block_hash not in recorded:
                recorded.add(block_hash)
                self._num_owners_by_hash[block_hash] = (
                    self._num_owners_by_hash.get(block_hash, 0) + 1
                )
        request.num_recorded_block_hashes = len(request.block_hashes)

    def _drop_owners(self, block_hashes: Iterable[BlockHash]) -> list[BlockHash]:
        """Drop one owner of each hash; return the hashes that no live session records anymore."""
        unowned = []
        for block_hash in block_hashes:
            num_owners = self._num_owners_by_hash.pop(block_hash) - 1
            if num_owners:
                self._num_owners_by_hash[block_hash] = num_owners
            else:
                unowned.append(block_hash)
        return unowned

    def _evict(self, block_hashes: list[BlockHash]) -> None:
        """Evict the free blocks cached under ``block_hashes`` that no live session records."""
        if not block_hashes:
            return
        self._record_in_flight_block_hashes()
        blocks = [
            block
            for block_hash in block_hashes
            for block in self._evict_hash(block_hash)
        ]
        # Block id order, so every TP rank leaves the same free queue.
        self._block_pool.free_block_queue.prepend_n(
            sorted(blocks, key=lambda block: block.block_id)
        )
        self.num_evicted_blocks += len(blocks)

    def _record_in_flight_block_hashes(self) -> None:
        # A preempted request's output blocks are free and cached until it resumes, and another
        # session may share them (a judge that read this rollout's reply). Count them first.
        for request in self._requests.values():
            if not request.session_released:
                self._record_new_block_hashes(request)

    def _evict_hash(self, block_hash: BlockHash) -> list[KVCacheBlock]:
        """Take the free blocks cached under ``block_hash`` out of the free queue and drop their
        hash, unless a live session records it (e.g. a preempted request's output)."""
        evicted = []
        for kv_cache_group_id in self._kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
            for block in self._cached_copies(key):
                if block.ref_cnt == 0 and not self._has_live_owner(block):
                    self._block_pool.free_block_queue.remove(block)
                    # Drops the block's hash, as vLLM does when it reuses a cached block.
                    self._block_pool.evict_blocks({block.block_id})
                    evicted.append(block)
        return evicted

    def _cached_copies(self, key: BlockHashWithGroupId) -> list[KVCacheBlock]:
        """Every block cached under ``key``. vLLM may cache one hash in several blocks, and its
        lookup shows one at a time: take each out of the map to see the next, then put all back."""
        cached_blocks = self._block_pool.cached_block_hash_to_block
        copies = []
        while (block := cached_blocks.get_one_block(key)) is not None:
            cached_blocks.pop(key, block.block_id)
            copies.append(block)
        for block in copies:
            cached_blocks.insert(key, block)
        return copies

    def _has_live_owner(self, block: KVCacheBlock) -> bool:
        return get_block_hash(block.block_hash) in self._num_owners_by_hash
