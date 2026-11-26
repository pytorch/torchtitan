# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evict ended multi-turn sessions' KV from vLLM's prefix cache."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TrackedRequest:
    """One request's block hashes, counted while they may still be reused."""

    internal_request_id: str
    vllm_request: Request
    """vLLM's request, read for its computed tokens."""
    block_hashes: list[BlockHash]
    """vLLM's `Request.block_hashes`, which vLLM extends in place as the reply fills blocks."""
    num_counted: int = 0
    """How many of `block_hashes` are counted in `_num_requests_by_hash`."""
    num_swept_tokens: int = 0
    """Computed tokens at the last sweep of this request's superseded GDN states."""
    gdn_scan_starts: list[int] = field(default_factory=list)
    """Per GDN group, the first index of `block_hashes` the next sweep looks up."""


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
    - GDN (align-mode Mamba) groups cache a state at each block boundary. A live session keeps
      one state per group, the one a next turn that extends the history hits; its other states
      are evicted unless another request counts their hash (a group's shared prompt):

        session B's states at blocks 3, 4, 5  -> 5 stays; 3 and 4 evicted
        B's next prompt rewrites block 4      -> resumes from the shared prompt's state
        B's last sibling ends, B is idle      -> the shared prompt's state goes too

    - Siblings that prefill one prompt in the same step each cache their own copy of its
      blocks, and with MTP drafts a turn caches its hit's last block again (vLLM doesn't
      de-duplicate). Once a request leaves vLLM, each of its blocks keeps one cached copy; the
      other free copies are evicted.

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
        #   class's reads and edits of KV cache manager internals.
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
        self._coordinator = kv_cache_manager.coordinator
        self._kv_cache_group_ids = range(kv_cache_manager.num_kv_cache_groups)
        gdn_groups = [
            (kv_cache_group_id, group.kv_cache_spec.block_size)
            for kv_cache_group_id, group in enumerate(
                kv_cache_manager.kv_cache_config.kv_cache_groups
            )
            if isinstance(group.kv_cache_spec, MambaSpec)
            and group.kv_cache_spec.mamba_cache_mode == "align"
        ]
        self._gdn_kv_cache_group_ids = [group_id for group_id, _ in gdn_groups]
        self._gdn_block_size = min((size for _, size in gdn_groups), default=0)
        # Live sessions' newest requests, until `after_step` sweeps them after they leave vLLM
        # and none of their superseded GDN states is in use.
        self._sessions_to_sweep: dict[str, _TrackedRequest] = {}
        # Superseded GDN states that sweeps kept because other requests counted them (a group's
        # prompt), by hash; `_forget` sweeps their sessions again once at most one request counts it.
        self._sessions_by_shared_gdn_state: dict[BlockHash, set[str]] = {}
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
        # Plain ints, so other threads (e.g. a logging endpoint) can read them safely. Blocks
        # of ended sessions and requests; and blocks live sessions no longer need: superseded
        # GDN states and extra copies of shared blocks.
        self.num_evicted_blocks = 0
        self.num_evicted_live_session_blocks = 0

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
        vllm_request = self._scheduler.requests[internal_request_id]
        request = _TrackedRequest(
            internal_request_id=internal_request_id,
            vllm_request=vllm_request,
            block_hashes=vllm_request.block_hashes,
            gdn_scan_starts=[0] * len(self._gdn_kv_cache_group_ids),
        )
        self._count_new_block_hashes(request)
        previous = self._sessions.get(session_id)
        self._sessions[session_id] = request
        self._group_sessions.setdefault(group_id, set()).add(session_id)
        self._sessions_to_sweep[session_id] = request
        if previous is not None:
            self._forget([previous])

    def after_step(self) -> None:
        """Evict the blocks of replaced or released requests that left vLLM in this step, the
        blocks that were in use at their eviction and are free now, live sessions'
        superseded GDN states, and the extra copies of blocks that requests leaving vLLM share."""
        if self._held_block_hashes:
            # A hash a request counts again stays: that request's end evicts it.
            self._count_all_new_block_hashes()
            self._held_block_hashes = self._evict(
                [
                    block_hash
                    for block_hash in self._held_block_hashes
                    if block_hash not in self._num_requests_by_hash
                ],
                kv_cache_group_ids=self._kv_cache_group_ids,
            )

        # Sweep a request when its computed tokens cross a GDN block, and once after it
        # leaves vLLM. A request still waiting has computed nothing, so its history's state stays.
        to_sweep: list[tuple[str, _TrackedRequest, bool]] = []
        block_size = self._gdn_block_size
        for session_id, request in self._sessions_to_sweep.items():
            num_computed_tokens = request.vllm_request.num_computed_tokens
            left_vllm = request.internal_request_id not in self._scheduler.requests
            crossed_gdn_block = bool(self._gdn_kv_cache_group_ids) and (
                num_computed_tokens // block_size
                > request.num_swept_tokens // block_size
            )
            if left_vllm or crossed_gdn_block:
                request.num_swept_tokens = max(
                    request.num_swept_tokens, num_computed_tokens
                )
                to_sweep.append((session_id, request, left_vllm))

        finished = [
            request
            for internal_id, request in self._ended_requests.items()
            if internal_id not in self._scheduler.requests
        ]
        if to_sweep or finished:
            self._count_all_new_block_hashes()
        for session_id, request, left_vllm in to_sweep:
            has_state_in_use = False
            if self._gdn_kv_cache_group_ids:
                has_state_in_use = self._evict_superseded_gdn_states(
                    session_id, request, left_vllm=left_vllm
                )
            if left_vllm:
                self._evict_duplicate_copies(request)
                # Swept again at the next step while a superseded state is in use.
                if not has_state_in_use:
                    del self._sessions_to_sweep[session_id]

        # Ended requests share their blocks with their session's next request or a sibling:
        # de-duplicate before uncounting them.
        for request in finished:
            self._evict_duplicate_copies(request)
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
        released_session_ids = [
            session_id
            for session_id in sorted(released)
            if session_id in self._sessions
        ]
        for session_id in released_session_ids:
            self._sessions_to_sweep.pop(session_id, None)
        for swept_session_ids in self._sessions_by_shared_gdn_state.values():
            swept_session_ids.difference_update(released_session_ids)
        self._forget(
            [self._sessions.pop(session_id) for session_id in released_session_ids]
        )
        self._release_cpu_cache()

    def _forget(self, requests: list[_TrackedRequest]) -> None:
        """Uncount `requests` that left vLLM and evict their hashes that nothing counts; keep
        counting the others until `after_step` sees them leave."""
        ended_block_hashes: list[BlockHash] = []
        sessions_to_sweep_again: set[str] = set()
        for request in requests:
            if request.internal_request_id in self._scheduler.requests:
                self._ended_requests[request.internal_request_id] = request
                continue
            self._count_new_block_hashes(request)
            for block_hash in request.block_hashes:
                num_requests = self._num_requests_by_hash.pop(block_hash) - 1
                if num_requests:
                    self._num_requests_by_hash[block_hash] = num_requests
                if (
                    num_requests <= 1
                    and block_hash in self._sessions_by_shared_gdn_state
                ):
                    sessions_to_sweep_again |= self._sessions_by_shared_gdn_state.pop(
                        block_hash
                    )
            ended_block_hashes += request.block_hashes

        # Sessions that kept a shared superseded GDN state, which at most one request counts now.
        # An idle one sweeps now: the engine may not step again before its next turn. A running
        # one sweeps when it leaves vLLM.
        if sessions_to_sweep_again:
            self._count_all_new_block_hashes()
        for session_id in sorted(sessions_to_sweep_again):
            session = self._sessions.get(session_id)
            if (
                session is None
                or session.internal_request_id in self._scheduler.requests
            ):
                continue
            if self._evict_superseded_gdn_states(session_id, session, left_vllm=True):
                self._sessions_to_sweep[session_id] = session

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
        self._held_block_hashes |= self._evict(
            released, kv_cache_group_ids=self._kv_cache_group_ids
        )
        if self._offload_connector is not None:
            self._cpu_hashes_to_release.update(released)

    def _evict_superseded_gdn_states(
        self, session_id: str, request: _TrackedRequest, *, left_vllm: bool
    ) -> bool:
        """Per GDN group, keep the state vLLM's prefix lookup picks for `request`'s tokens, which
        its next turn resumes from. Evict the older ones that no other request counts, and once
        it left vLLM, the newer ones too. Return whether a state to evict was in use.

            no drafts:            states at blocks 1, 2, 3 -> keeps 3
            MTP's block drop on:  states at blocks 0, 2, 3 -> keeps 2; 3 only while it runs
        """
        # The lookup applies MTP's block drop, `prefix_match_unit` and vLLM's alignment.
        hit_blocks, _, _ = self._coordinator.find_longest_cache_hit(
            request.block_hashes, request.vllm_request.num_tokens
        )
        get_one_block = self._block_pool.cached_block_hash_to_block.get_one_block
        block_hashes = request.block_hashes
        has_state_in_use = False
        for position, kv_cache_group_id in enumerate(self._gdn_kv_cache_group_ids):
            kept_keys = {
                block.block_hash
                for block in hit_blocks[kv_cache_group_id]
                if not block.is_null
            }
            cached_indices, kept_index = [], None
            for index in range(request.gdn_scan_starts[position], len(block_hashes)):
                key = make_block_hash_with_group_id(
                    block_hashes[index], kv_cache_group_id
                )
                if get_one_block(key) is None:
                    continue
                cached_indices.append(index)
                if key in kept_keys:
                    kept_index = index
            if kept_index is None:
                continue

            # A superseded state that other requests count stays, recorded for `_forget`.
            superseded_indices, shared_indices = [], []
            for index in cached_indices:
                if index == kept_index or (index > kept_index and not left_vllm):
                    continue
                block_hash = block_hashes[index]
                if self._num_requests_by_hash[block_hash] == 1:
                    superseded_indices.append(index)
                else:
                    shared_indices.append(index)
                    self._sessions_by_shared_gdn_state.setdefault(
                        block_hash, set()
                    ).add(session_id)
            held_block_hashes = self._evict(
                [block_hashes[index] for index in superseded_indices],
                kv_cache_group_ids=(kv_cache_group_id,),
                live_session=True,
            )
            # A copy still in use (the request's previous state, freed next step) and a shared
            # state are looked up again in a later sweep, which starts at or before them.
            held_indices = [
                index
                for index in superseded_indices
                if block_hashes[index] in held_block_hashes
            ]
            request.gdn_scan_starts[position] = min(
                [kept_index, *held_indices, *shared_indices]
            )
            has_state_in_use = has_state_in_use or bool(held_indices)
        return has_state_in_use

    def _evict_duplicate_copies(self, request: _TrackedRequest) -> None:
        """For each block of `request`, keep one cached copy per KV cache group and evict its
        other free copies:

            siblings A, B prefill prompt P in one step     -> P's blocks cached twice: A's, B's
            A leaves vLLM, B still runs                    -> A's goes; B's stays (in use)
            B leaves vLLM, a KV connector's save holds it  -> B's stays, the one copy
            MTP drafts: a turn recomputes its hit's last   -> the copy lookups return stays
            block and caches it again

        If two copies are still in use (two KV connector saves) when the last sharer leaves, both
        stay until a sibling's next turn leaves vLLM or the group is released.
        """
        self._evict(
            request.block_hashes,
            kv_cache_group_ids=self._kv_cache_group_ids,
            keep_one_copy=True,
            live_session=True,
        )

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

    def _evict(
        self,
        block_hashes: list[BlockHash],
        *,
        kv_cache_group_ids: Sequence[int],
        keep_one_copy: bool = False,
        live_session: bool = False,
    ) -> set[BlockHash]:
        """Move the free blocks of `block_hashes` in `kv_cache_group_ids` to the free queue's
        front, without their hash; return the hashes that kept a copy in use. With
        `keep_one_copy`, one copy stays cached: a copy in use if there is one, else the copy
        prefix lookups return first. `live_session` counts them in
        `num_evicted_live_session_blocks`, not `num_evicted_blocks`."""
        cached_blocks = self._block_pool.cached_block_hash_to_block
        free_block_queue = self._block_pool.free_block_queue
        evicted = []
        held_block_hashes = set()
        for block_hash in block_hashes:
            for kv_cache_group_id in kv_cache_group_ids:
                key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
                # vLLM may cache one hash in several blocks (siblings that prefilled one
                # prompt in the same step). Each lookup returns one, so set each aside to
                # reach the next, then put them back in order.
                copies = []
                while (block := cached_blocks.get_one_block(key)) is not None:
                    cached_blocks.pop(key, block.block_id)
                    copies.append(block)
                for block in copies:
                    cached_blocks.insert(key, block)

                # A copy in use (a running request, a KV connector's save or lazy copy, a free
                # that async scheduling defers) keeps its hash, so it is the copy that stays.
                free_copies = [block for block in copies if block.ref_cnt == 0]
                if len(free_copies) < len(copies):
                    held_block_hashes.add(block_hash)
                elif keep_one_copy:
                    free_copies = free_copies[1:]
                for block in free_copies:
                    free_block_queue.remove(block)
                    # Drops the block's hashes, as vLLM does when it reuses a cached block.
                    self._block_pool.evict_blocks({block.block_id})
                    evicted.append(block)
        # Block id order, so every TP rank leaves the same free queue.
        free_block_queue.prepend_n(sorted(evicted, key=lambda block: block.block_id))
        if live_session:
            self.num_evicted_live_session_blocks += len(evicted)
        else:
            self.num_evicted_blocks += len(evicted)
        return held_block_hashes
