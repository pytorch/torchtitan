# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep each multi-turn session's reusable KV cached between its turns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

# A session's mark on one block: the block and its reuse epoch when marked (see `_TieredFreeBlockQueue`).
BlockMark = tuple[KVCacheBlock, int]


class _TieredFreeBlockQueue:
    """vLLM's free block queue plus an evict-last tier for blocks that live sessions will reuse.

    Stands in for the ``FreeKVCacheBlockQueue`` of vLLM's ``BlockPool`` and implements what the pool
    calls after its init: ``popleft_n``, ``remove``, ``append_n``, ``prepend_n`` and
    ``num_free_blocks``. A free block sits in the live tier while at least one live session marks
    it; vLLM reuses live-tier blocks, oldest first, only once the ordinary tier is empty. Both tiers
    count as free, so vLLM's admission sees every block it can reclaim.

    Args:
        ordinary: The block pool's own queue; it becomes the ordinary (LRU) tier.
    """

    def __init__(self, ordinary: FreeKVCacheBlockQueue) -> None:
        self._ordinary = ordinary
        self._live = FreeKVCacheBlockQueue([])
        # Live-session marks per block id. A free block is in the live tier iff it has marks.
        self._marks: dict[int, int] = {}
        # Bumped each time vLLM reuses a marked block, so that marks taken on its old content are
        # not dropped against its new content.
        self._epochs: dict[int, int] = {}
        # Plain int, so other threads (e.g. a logging endpoint) can read it safely.
        self.num_reused_blocks = 0

    @property
    def num_free_blocks(self) -> int:
        return self._ordinary.num_free_blocks + self._live.num_free_blocks

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        num_ordinary = min(n, self._ordinary.num_free_blocks)
        blocks = self._ordinary.popleft_n(num_ordinary)
        if n > num_ordinary:
            reused = self._live.popleft_n(n - num_ordinary)
            for block in reused:
                # vLLM overwrites the block, so the sessions' marks on its old content are gone.
                del self._marks[block.block_id]
                self._epochs[block.block_id] = self._epochs.get(block.block_id, 0) + 1
            self.num_reused_blocks += len(reused)
            blocks += reused
        return blocks

    def remove(self, block: KVCacheBlock) -> None:
        self._tier(block).remove(block)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        live, ordinary = self._split(blocks)
        self._ordinary.append_n(ordinary)
        self._live.append_n(live)

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        live, ordinary = self._split(blocks)
        self._ordinary.prepend_n(ordinary)
        self._live.prepend_n(live)

    def mark(self, blocks: Sequence[KVCacheBlock]) -> list[BlockMark]:
        """Mark blocks as reused by a live session; free ones move to the live tier.

        Returns:
            The marks, to pass to `unmark` once the session no longer reuses the blocks.
        """
        marks = []
        for block in blocks:
            count = self._marks.get(block.block_id, 0)
            if count == 0 and block.ref_cnt == 0:
                self._ordinary.remove(block)
                self._live.append_n([block])
            self._marks[block.block_id] = count + 1
            marks.append((block, self._epochs.get(block.block_id, 0)))
        return marks

    def unmark(self, marks: Sequence[BlockMark]) -> None:
        """Drop marks taken by `mark`; free blocks left unmarked go to the evict-first end."""
        unmarked = []
        for block, epoch in marks:
            if self._epochs.get(block.block_id, 0) != epoch:
                continue  # vLLM reused the block after this mark was taken
            count = self._marks[block.block_id] - 1
            if count > 0:
                self._marks[block.block_id] = count
                continue
            del self._marks[block.block_id]
            if block.ref_cnt == 0:
                self._live.remove(block)
                unmarked.append(block)
        # The session's next turn moved past these blocks, or the session ended. Reversed, so the
        # tail of the prefix goes first (vLLM frees a request's blocks in the same order).
        self._ordinary.prepend_n(unmarked[::-1])

    def _tier(self, block: KVCacheBlock) -> FreeKVCacheBlockQueue:
        return self._live if block.block_id in self._marks else self._ordinary

    def _split(
        self, blocks: list[KVCacheBlock]
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        if not self._marks:
            return [], blocks
        live = [block for block in blocks if block.block_id in self._marks]
        ordinary = [block for block in blocks if block.block_id not in self._marks]
        return live, ordinary


class SessionKVHolder:
    """Keeps a session's reusable prefix cached until its next turn or its end.

    vLLM takes the oldest free block whenever it needs a new one, so between a session's turns its
    history competes on age with every other free block, including blocks of rollouts that already
    ended. In vLLM's hybrid "align" mode the prompt-end Gated-DeltaNet state even becomes free when
    decode starts. So this swaps the block pool's free queue for a two-tier one
    (`_TieredFreeBlockQueue`) and, right after a request's prefill, marks the blocks vLLM's own prefix
    lookup says a continuation would hit (attention blocks and the matching GDN state blocks). vLLM
    reuses a marked block only once no unmarked free block is left. Marked blocks still count as
    free, so they never keep vLLM from admitting a request.

    A session's marks move to its next turn's prefix, and are dropped when the session or its group
    ends; free blocks no live session marks go to the evict-first end of the queue. Every mutating
    method must run on the engine thread of every rank, in the same order, so the schedulers of all
    TP ranks stay identical.

    Example:

        holder = SessionKVHolder(scheduler)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        holder.track(internal_id, session_id="group=3/rollout=0", group_id=3)
        engine.step()
        holder.after_step()  # marks the prefix of requests whose prefill just finished
        holder.release(session_ids=["group=3/rollout=0"])  # the rollout ended

    Args:
        scheduler: The vLLM v1 scheduler of this rank's in-process engine.
    """

    def __init__(self, scheduler: Any) -> None:
        kv_cache_manager = scheduler.kv_cache_manager
        block_pool = kv_cache_manager.block_pool
        if scheduler.connector is not None:
            raise ValueError(
                "hold_session_kv does not support KV connectors: they manage block lifetimes too"
            )
        if any(
            manager.block_pool is not block_pool
            for manager in kv_cache_manager.coordinator.single_type_managers
        ):
            raise ValueError(
                "hold_session_kv supports only models whose KV cache groups share one block pool"
            )
        if type(block_pool.free_block_queue) is not FreeKVCacheBlockQueue:
            raise ValueError(
                f"hold_session_kv expects vLLM's FreeKVCacheBlockQueue, "
                f"got {type(block_pool.free_block_queue).__name__}"
            )
        self._scheduler = scheduler
        self._coordinator = kv_cache_manager.coordinator
        self._queue = _TieredFreeBlockQueue(block_pool.free_block_queue)
        block_pool.free_block_queue = self._queue
        # Requests in vLLM: internal request id -> (session, group, prefix marked).
        self._live: dict[str, tuple[str, int, bool]] = {}
        # Marks per session on the prefix its next turn would hit.
        self._session_marks: dict[str, list[BlockMark]] = {}
        self._session_groups: dict[str, int] = {}
        self._group_sessions: dict[int, set[str]] = {}
        # Plain int, so other threads (e.g. a logging endpoint) can read it safely.
        self.num_sessions = 0

    @property
    def num_reused_blocks(self) -> int:
        """Marked blocks vLLM reused because no unmarked free block was left."""
        return self._queue.num_reused_blocks

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Mark this request's prefix once its prefill finishes."""
        self._live[internal_request_id] = (session_id, group_id, False)

    def after_step(self) -> None:
        """Mark the prefix of requests whose prefill finished; forget requests that left vLLM."""
        for internal_id, (session_id, group_id, marked) in list(self._live.items()):
            request = self._scheduler.requests.get(internal_id)
            if request is None:  # finished or aborted
                del self._live[internal_id]
            elif (
                not marked and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                self._live[internal_id] = (session_id, group_id, True)
                self._mark(request, session_id, group_id)

    def release(
        self, session_ids: Sequence[str] = (), group_ids: Sequence[int] = ()
    ) -> None:
        """Unmark sessions that make no more calls, and every session of finished groups."""
        released = set(session_ids)
        for group_id in group_ids:
            released |= self._group_sessions.pop(group_id, set())
        # Sorted, not set order: string hashes differ per process, and every TP rank must reorder
        # its free queue the same way.
        for session_id in sorted(released):
            self._unmark(session_id)
        # A request still before its prefill must not mark a released session again.
        dropped_groups = set(group_ids)
        for internal_id, (session_id, group_id, marked) in list(self._live.items()):
            if session_id in released or group_id in dropped_groups:
                self._live[internal_id] = (session_id, group_id, True)

    def _mark(self, request: Any, session_id: str, group_id: int) -> None:
        # The prompt's last token is recomputed for its logits, so a continuation hits at most
        # num_prompt_tokens - 1 tokens; the lookup returns the attention blocks and, per GDN
        # group, the state block that ends on that aligned boundary. vLLM returns the oldest
        # cached copy of each block, which is also the copy the continuation's lookup will hit.
        hit_blocks, _, _ = self._coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_prompt_tokens - 1
        )
        blocks = [b for kv_group in hit_blocks for b in kv_group if not b.is_null]
        # Mark the new prefix before unmarking the old one, so shared blocks stay in the live tier.
        marks = self._queue.mark(blocks)
        self._unmark(session_id)
        self._session_marks[session_id] = marks
        self._session_groups[session_id] = group_id
        self._group_sessions.setdefault(group_id, set()).add(session_id)
        self.num_sessions += 1

    def _unmark(self, session_id: str) -> None:
        marks = self._session_marks.pop(session_id, None)
        group_id = self._session_groups.pop(session_id, None)
        if group_id is not None:
            self._group_sessions.get(group_id, set()).discard(session_id)
        if marks is not None:
            self.num_sessions -= 1
            self._queue.unmark(marks)
